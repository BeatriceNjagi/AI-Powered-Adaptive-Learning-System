import asyncpg
import os
import asyncio
import aiohttp
import json
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Mixtral was decommissioned by Groq - llama3-8b-8192 is a reliable free replacement
# You can also use "llama3-70b-8192" for smarter answers (slower)
GROQ_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


async def build_feedback_prompt(
    topic: str,
    question: str,
    student_answer: str,
    correct_answer: str,
    is_correct: bool,
) -> str:
    """Constructs a text prompt suitable for Groq AI.

    Groq cannot see your database, so every piece of context must be
    included in the prompt. The prompt changes based on whether the student
    was right or wrong.
    """
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


async def get_ai_feedback(
    topic: str,
    question: str,
    student_answer: str,
    correct_answer: str,
    is_correct: bool,
) -> str:
    """
    Calls Groq to generate feedback for a student's answer.

    Steps:
    1. Build a context-aware prompt
    2. Send to Groq API with a timeout
    3. Parse and return the response
    4. If Groq fails for any reason, return fallback feedback so the
       quiz result can still be saved to the database
    """
    prompt = await build_feedback_prompt(
        topic=topic,
        question=question,
        student_answer=student_answer,
        correct_answer=correct_answer,
        is_correct=is_correct,
    )

    # If no API key is set, just return a basic fallback
    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY not set, using fallback feedback")
        return generate_fallback_feedback(is_correct, correct_answer)

    try:
        feedback = await call_groq(prompt)
        return feedback
    except asyncio.TimeoutError:
        print("Groq timed out - using fallback feedback")
        return generate_fallback_feedback(
            is_correct, correct_answer, reason="(AI response timed out)"
        )
    except aiohttp.ClientError as e:
        print(f"Groq network error: {e}")
        return generate_fallback_feedback(
            is_correct, correct_answer, reason="(AI service unavailable)"
        )
    except Exception as e:
        print(f"Unexpected Groq error: {e}")
        return generate_fallback_feedback(is_correct, correct_answer)


async def call_groq(prompt: str, timeout_seconds: int = 15) -> str:
    """
    Makes an HTTP POST request to Groq's API.

    Uses aiohttp for async HTTP calls and includes a timeout so the
    app doesn't hang forever waiting for a response.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    model = os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a friendly and helpful tutor.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.7,
        "max_tokens": 500,
    }

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(GROQ_API_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"Groq error {resp.status}: {error_text}")
                raise Exception(f"Groq API returned status {resp.status}: {error_text}")

            data = await resp.json()
            # Groq responds in the same format as OpenAI
            return data["choices"][0]["message"]["content"]


def generate_fallback_feedback(
    is_correct: bool,
    correct_answer: str,
    reason: str = "",
) -> str:
    """
    Simple fallback feedback when Groq is not available.

    This makes sure students always get some kind of response even if
    the AI service is down.
    """
    if is_correct:
        feedback = "Correct! Well done, keep it up."
    else:
        feedback = f"Not quite right. The correct answer is: {correct_answer}. Review the material and try again!"

    if reason:
        feedback += f" {reason}"

    return feedback


async def generate_quiz_question(topic: str, difficulty: str) -> dict:
    """
    Asks Groq to generate a multiple choice question and returns it as a dict.

    We put the strict JSON instruction in the system prompt so Groq follows
    it more reliably. We also log every step so if something goes wrong you
    can see exactly what Groq returned in the terminal.
    """

    # Return a placeholder if no API key is set
    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY not set in .env - returning placeholder question")
        return {
            "question": f"Sample question about {topic} ({difficulty})",
            "correct_answer": "Answer A",
            "choices": ["Answer A", "Answer B", "Answer C", "Answer D"],
        }

    # Put the JSON format rule in the system prompt - Groq follows system
    # instructions more reliably than user instructions for structured output
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
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 400,
        }

        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(GROQ_API_URL, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"Groq API error {resp.status}: {error_text}")
                    raise Exception(f"Groq returned status {resp.status}")

                data = await resp.json()
                response_text = data["choices"][0]["message"]["content"]

        print(f"Groq raw response: {response_text}")

        # Strip markdown fences if Groq added them despite instructions
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]  # drop the ```json line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # drop the closing ```
            cleaned = "\n".join(lines).strip()

        question_data = json.loads(cleaned)
        print(f"Parsed question data: {question_data}")

        # Validate all required fields exist
        if "question" not in question_data:
            raise ValueError("Missing 'question' field in Groq response")
        if "correct_answer" not in question_data:
            raise ValueError("Missing 'correct_answer' field in Groq response")
        if "choices" not in question_data or not question_data["choices"]:
            raise ValueError("Missing 'choices' field in Groq response")

        choices = question_data["choices"]

        # Make sure we have at least 2 choices
        if len(choices) < 2:
            raise ValueError(f"Not enough choices returned: {choices}")

        # Safety check - make sure correct answer is actually in the choices
        if question_data["correct_answer"] not in choices:
            print(f"Warning: correct answer not in choices, appending it")
            choices.append(question_data["correct_answer"])

        print(f"Question generated successfully: {question_data['question'][:50]}...")
        return {
            "question": question_data["question"],
            "correct_answer": question_data["correct_answer"],
            "choices": choices,
        }

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response that failed to parse: {response_text}")
        raise Exception(f"Groq returned invalid JSON: {e}")

    except Exception as e:
        print(f"Question generation failed: {e}")
        raise


async def save_question(
    conn,
    topic: str,
    difficulty: str,
    question: str,
    correct_answer: str,
    choices: list | None = None,
) -> int:
    """Saves a generated question to the database and returns its ID."""
    choices_json = json.dumps(choices) if choices else None

    query = """
    INSERT INTO questions (topic, difficulty, question, correct_answer, choices, created_at)
    VALUES ($1, $2, $3, $4, $5, NOW())
    RETURNING id
    """
    question_id = await conn.fetchval(
        query, topic, difficulty, question, correct_answer, choices_json
    )
    return question_id


async def get_random_question(conn, topic: str, difficulty: str = None) -> dict | None:
    """Gets a random stored question from the database for a given topic."""
    if difficulty:
        query = """
        SELECT id, topic, difficulty, question, correct_answer, choices, created_at
        FROM questions
        WHERE topic = $1 AND difficulty = $2
        ORDER BY RANDOM()
        LIMIT 1
        """
        row = await conn.fetchrow(query, topic, difficulty)
    else:
        query = """
        SELECT id, topic, difficulty, question, correct_answer, choices, created_at
        FROM questions
        WHERE topic = $1
        ORDER BY RANDOM()
        LIMIT 1
        """
        row = await conn.fetchrow(query, topic)

    if row:
        result = dict(row)
        result["choices"] = json.loads(result["choices"]) if result["choices"] else None
        return result
    return None


async def get_questions_by_topic(conn, topic: str, limit: int = 10) -> list[dict]:
    """Gets all stored questions for a specific topic."""
    query = """
    SELECT id, topic, difficulty, question, correct_answer, choices, created_at
    FROM questions
    WHERE topic = $1
    ORDER BY created_at DESC
    LIMIT $2
    """
    rows = await conn.fetch(query, topic, limit)
    results = []
    for row in rows:
        result = dict(row)
        result["choices"] = json.loads(result["choices"]) if result["choices"] else None
        results.append(result)
    return results


async def save_quiz_result(conn, data, is_correct: bool, feedback: str):
    """
    Saves a quiz result to the database.
    Also updates the performance_history table so we always have
    up-to-date accuracy summaries per student per topic.
    """
    query = """
    INSERT INTO quiz_results (
        student_id, topic, question, student_answer,
        correct_answer, is_correct, time_taken, feedback, submitted_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
    """
    await conn.execute(
        query,
        data.student_id,
        data.topic,
        data.question,
        data.student_answer,
        data.correct_answer,
        is_correct,
        data.time_taken,
        feedback,
    )

    # Update the performance summary table so analytics stay current
    await update_performance_history(conn, data.student_id, data.topic)


async def get_student_progress(conn, student_id: int) -> list[dict]:
    """
    Gets a student's quiz results grouped by topic.
    Returns total attempts, correct attempts, and accuracy percentage.
    """
    query = """
    SELECT
        topic,
        COUNT(*) as total_attempts,
        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_attempts,
        ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_percentage
    FROM quiz_results
    WHERE student_id = $1
    GROUP BY topic
    ORDER BY topic
    """
    rows = await conn.fetch(query, student_id)
    return [dict(row) for row in rows]


async def get_struggling_students(conn, score_threshold: float) -> list[dict]:
    """
    Finds students whose overall accuracy is below the given threshold.

    Uses HAVING instead of WHERE because we're filtering on an aggregated
    value (average accuracy), not on individual rows.
    """
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
    rows = await conn.fetch(query, score_threshold)
    return [dict(row) for row in rows]


async def get_hardest_topic(conn) -> list[dict]:
    """
    Returns topics sorted by lowest average accuracy across all students.
    Teachers can use this to see which topics need more curriculum support.
    """
    query = """
    SELECT
        topic,
        COUNT(*) as total_attempts,
        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_attempts,
        ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as average_accuracy
    FROM quiz_results
    GROUP BY topic
    ORDER BY average_accuracy ASC
    """
    rows = await conn.fetch(query)
    return [dict(row) for row in rows]


async def get_individual_student_report(conn, student_id: int) -> list[dict]:
    """
    Breaks down a single student's performance by topic.
    Sorted weakest to strongest so the hardest areas are easy to spot.
    """
    query = """
    SELECT
        student_id,
        topic,
        COUNT(*) as total_attempts,
        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_attempts,
        ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_percentage
    FROM quiz_results
    WHERE student_id = $1
    GROUP BY student_id, topic
    ORDER BY accuracy_percentage ASC
    """
    rows = await conn.fetch(query, student_id)
    return [dict(row) for row in rows]


# ── Student management ────────────────────────────────────────────────────

async def create_student(conn, name: str, email: str) -> int:
    """Creates a new student record and returns their ID."""
    query = """
    INSERT INTO students (name, email, joined_at)
    VALUES ($1, $2, NOW())
    RETURNING id
    """
    student_id = await conn.fetchval(query, name, email)
    return student_id


async def get_student(conn, student_id: int) -> dict | None:
    """Gets a single student by their ID."""
    query = "SELECT id, name, email, joined_at FROM students WHERE id = $1"
    row = await conn.fetchrow(query, student_id)
    return dict(row) if row else None


async def get_all_students(conn) -> list[dict]:
    """Gets all students, newest first."""
    query = "SELECT id, name, email, joined_at FROM students ORDER BY joined_at DESC"
    rows = await conn.fetch(query)
    return [dict(row) for row in rows]


# ── Performance history (summary cache) ──────────────────────────────────

async def update_performance_history(conn, student_id: int, topic: str):
    """
    Updates the performance_history table for a student-topic pair.

    Uses INSERT ... ON CONFLICT so it creates the row the first time
    and updates it on every subsequent quiz submission.
    """
    query = """
    INSERT INTO performance_history (student_id, topic, total_attempts, correct_attempts, average_accuracy, last_updated)
    SELECT
        student_id,
        topic,
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


async def get_performance_history(conn, student_id: int) -> list[dict]:
    """Gets the cached performance summary for a student, all topics."""
    query = """
    SELECT student_id, topic, total_attempts, correct_attempts, average_accuracy, last_updated
    FROM performance_history
    WHERE student_id = $1
    ORDER BY average_accuracy ASC
    """
    rows = await conn.fetch(query, student_id)
    return [dict(row) for row in rows]
