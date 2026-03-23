import asyncpg
import os
import asyncio
import aiohttp
import json
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


async def build_feedback_prompt(topic, question, student_answer, correct_answer, is_correct):
    """Builds a prompt for Groq based on whether the answer was correct."""
    if is_correct:
        return (
            f"You are a friendly tutor. "
            f"The student answered correctly on the topic '{topic}'.\n"
            f"Question: {question}\n"
            f"Student's answer: {student_answer}\n"
            f"Correct answer: {correct_answer}\n"
            "Provide a short congratulatory message and suggest a slightly "
            "more challenging follow-up question on the same topic."
        )
    else:
        return (
            f"You are a patient tutor. "
            f"The student got a question wrong on the topic '{topic}'.\n"
            f"Question: {question}\n"
            f"Student's answer: {student_answer}\n"
            f"Correct answer: {correct_answer}\n"
            "Explain why the student's answer is incorrect, walk through the "
            "reasoning with an example, and then ask the student to try a "
            "similar question to reinforce the concept."
        )


async def get_ai_feedback(topic, question, student_answer, correct_answer, is_correct):
    """Calls Groq for feedback. Falls back gracefully on any error."""
    prompt = await build_feedback_prompt(topic, question, student_answer, correct_answer, is_correct)

    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY not set, using fallback feedback")
        return generate_fallback_feedback(is_correct, correct_answer)

    try:
        return await call_groq(prompt)
    except asyncio.TimeoutError:
        print("Groq timed out - using fallback feedback")
        return generate_fallback_feedback(is_correct, correct_answer, reason="(AI response timed out)")
    except aiohttp.ClientError as e:
        print(f"Groq network error: {e}")
        return generate_fallback_feedback(is_correct, correct_answer, reason="(AI service unavailable)")
    except Exception as e:
        print(f"Unexpected Groq error: {e}")
        return generate_fallback_feedback(is_correct, correct_answer)


async def call_groq(prompt, timeout_seconds=15):
    """Makes an async POST request to the Groq API."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL),
        "messages": [
            {"role": "system", "content": "You are a friendly and helpful tutor."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 500,
    }

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(GROQ_API_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Groq API returned status {resp.status}: {error_text}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


def generate_fallback_feedback(is_correct, correct_answer, reason=""):
    """Returns basic feedback when Groq is unavailable."""
    if is_correct:
        feedback = "Correct! Well done, keep it up."
    else:
        feedback = f"Not quite right. The correct answer is: {correct_answer}. Review the material and try again!"
    if reason:
        feedback += f" {reason}"
    return feedback


async def generate_quiz_question(topic, difficulty):
    """Asks Groq to generate a multiple-choice question. Returns a dict."""
    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY not set — returning placeholder question")
        return {
            "question": f"Sample question about {topic} ({difficulty})",
            "correct_answer": "Answer A",
            "choices": ["Answer A", "Answer B", "Answer C", "Answer D"],
        }

    system_prompt = (
        "You are a quiz question generator. "
        "You ONLY respond with raw valid JSON. "
        "No markdown, no code blocks, no explanation - just the JSON object."
    )

    user_prompt = (
        f"Generate a {difficulty} difficulty multiple choice question about '{topic}'.\n"
        "Return a JSON object with exactly these fields:\n"
        "  question: the question text (string)\n"
        "  correct_answer: the correct answer (string, must match one of the choices exactly)\n"
        "  choices: array of exactly 4 strings (the correct answer plus 3 wrong answers, shuffled)\n"
        "Example: "
        + json.dumps({
            "question": "What is the powerhouse of the cell?",
            "correct_answer": "Mitochondria",
            "choices": ["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"]
        })
    )

    try:
        print(f"Calling Groq for question: topic='{topic}' difficulty='{difficulty}'")

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 400,
        }

        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(GROQ_API_URL, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Groq returned status {resp.status}")
                data = await resp.json()
                response_text = data["choices"][0]["message"]["content"]

        print(f"Groq raw response: {response_text}")

        # Strip markdown fences if present
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        question_data = json.loads(cleaned)
        print(f"Parsed question data: {question_data}")

        if "question" not in question_data:
            raise ValueError("Missing 'question' field in Groq response")
        if "correct_answer" not in question_data:
            raise ValueError("Missing 'correct_answer' field in Groq response")
        if "choices" not in question_data or not question_data["choices"]:
            raise ValueError("Missing 'choices' field in Groq response")

        choices = question_data["choices"]
        if len(choices) < 2:
            raise ValueError(f"Not enough choices returned: {choices}")

        if question_data["correct_answer"] not in choices:
            print("Warning: correct answer not in choices, appending it")
            choices.append(question_data["correct_answer"])

        print(f"Question generated successfully: {question_data['question'][:50]}...")
        return {
            "question": question_data["question"],
            "correct_answer": question_data["correct_answer"],
            "choices": choices,
        }

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        raise Exception(f"Groq returned invalid JSON: {e}")
    except Exception as e:
        print(f"Question generation failed: {e}")
        raise


async def save_question(conn, topic, difficulty, question, correct_answer, choices=None):
    """Saves a generated question to the database and returns its ID."""
    choices_json = json.dumps(choices) if choices else None
    query = """
    INSERT INTO questions (topic, difficulty, question, correct_answer, choices, created_at)
    VALUES ($1, $2, $3, $4, $5, NOW())
    RETURNING id
    """
    return await conn.fetchval(query, topic, difficulty, question, correct_answer, choices_json)


async def get_random_question(conn, topic, difficulty=None):
    """Returns a random stored question for the given topic (and optionally difficulty)."""
    if difficulty:
        query = """
        SELECT id, topic, difficulty, question, correct_answer, choices, created_at
        FROM questions WHERE topic = $1 AND difficulty = $2
        ORDER BY RANDOM() LIMIT 1
        """
        row = await conn.fetchrow(query, topic, difficulty)
    else:
        query = """
        SELECT id, topic, difficulty, question, correct_answer, choices, created_at
        FROM questions WHERE topic = $1
        ORDER BY RANDOM() LIMIT 1
        """
        row = await conn.fetchrow(query, topic)

    if row:
        result = dict(row)
        result["choices"] = json.loads(result["choices"]) if result["choices"] else None
        return result
    return None


async def get_questions_by_topic(conn, topic, limit=10):
    """Returns stored questions for a topic, newest first."""
    query = """
    SELECT id, topic, difficulty, question, correct_answer, choices, created_at
    FROM questions WHERE topic = $1
    ORDER BY created_at DESC LIMIT $2
    """
    rows = await conn.fetch(query, topic, limit)
    results = []
    for row in rows:
        result = dict(row)
        result["choices"] = json.loads(result["choices"]) if result["choices"] else None
        results.append(result)
    return results


async def save_quiz_result(conn, data, is_correct, feedback):
    """Saves a quiz result and updates the performance summary table."""
    query = """
    INSERT INTO quiz_results (
        student_id, topic, question, student_answer,
        correct_answer, is_correct, time_taken, feedback, submitted_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
    """
    await conn.execute(
        query,
        data.student_id, data.topic, data.question, data.student_answer,
        data.correct_answer, is_correct, data.time_taken, feedback,
    )
    await update_performance_history(conn, data.student_id, data.topic)


async def get_student_progress(conn, student_id):
    """Returns per-topic accuracy stats for a student."""
    query = """
    SELECT
        topic,
        COUNT(*) as total_attempts,
        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_attempts,
        ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_percentage
    FROM quiz_results
    WHERE student_id = $1
    GROUP BY topic ORDER BY topic
    """
    return [dict(row) for row in await conn.fetch(query, student_id)]


async def get_struggling_students(conn, score_threshold):
    """Returns students whose overall accuracy is below the threshold."""
    query = """
    SELECT
        student_id,
        COUNT(*) as total_attempts,
        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_attempts,
        ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as average_accuracy
    FROM quiz_results
    GROUP BY student_id
    HAVING ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) < $1
    ORDER BY average_accuracy ASC
    """
    return [dict(row) for row in await conn.fetch(query, score_threshold)]


async def get_hardest_topic(conn):
    """Returns topics sorted by lowest average accuracy."""
    query = """
    SELECT
        topic,
        COUNT(*) as total_attempts,
        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_attempts,
        ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as average_accuracy
    FROM quiz_results
    GROUP BY topic ORDER BY average_accuracy ASC
    """
    return [dict(row) for row in await conn.fetch(query)]


async def get_individual_student_report(conn, student_id):
    """Returns per-topic breakdown for a student, sorted weakest first."""
    query = """
    SELECT
        student_id, topic,
        COUNT(*) as total_attempts,
        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_attempts,
        ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_percentage
    FROM quiz_results
    WHERE student_id = $1
    GROUP BY student_id, topic
    ORDER BY accuracy_percentage ASC
    """
    return [dict(row) for row in await conn.fetch(query, student_id)]


# ── Student management ────────────────────────────────────────────────────

async def create_student(conn, name, email):
    """Creates a student and returns their new ID."""
    query = "INSERT INTO students (name, email, joined_at) VALUES ($1, $2, NOW()) RETURNING id"
    return await conn.fetchval(query, name, email)


async def get_student(conn, student_id):
    """Returns a single student by ID, or None."""
    row = await conn.fetchrow("SELECT id, name, email, joined_at FROM students WHERE id = $1", student_id)
    return dict(row) if row else None


async def get_all_students(conn):
    """Returns all students, newest first."""
    rows = await conn.fetch("SELECT id, name, email, joined_at FROM students ORDER BY joined_at DESC")
    return [dict(row) for row in rows]


# ── Performance history ───────────────────────────────────────────────────

async def update_performance_history(conn, student_id, topic):
    """Upserts the performance summary for a student-topic pair."""
    query = """
    INSERT INTO performance_history (student_id, topic, total_attempts, correct_attempts, average_accuracy, last_updated)
    SELECT
        student_id, topic,
        COUNT(*) as total_attempts,
        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_attempts,
        ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as average_accuracy,
        NOW() as last_updated
    FROM quiz_results
    WHERE student_id = $1 AND topic = $2
    GROUP BY student_id, topic
    ON CONFLICT (student_id, topic)
    DO UPDATE SET
        total_attempts = EXCLUDED.total_attempts,
        correct_attempts = EXCLUDED.correct_attempts,
        average_accuracy = EXCLUDED.average_accuracy,
        last_updated = EXCLUDED.last_updated
    """
    await conn.execute(query, student_id, topic)


async def get_performance_history(conn, student_id):
    """Returns the cached performance summary for a student across all topics."""
    query = """
    SELECT student_id, topic, total_attempts, correct_attempts, average_accuracy, last_updated
    FROM performance_history
    WHERE student_id = $1
    ORDER BY average_accuracy ASC
    """
    return [dict(row) for row in await conn.fetch(query, student_id)]
