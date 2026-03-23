import asyncpg
from fastapi import FastAPI, HTTPException
from database import get_connection
from crud import (
    save_quiz_result,
    get_ai_feedback,
    get_student_progress,
    generate_quiz_question,
    get_struggling_students,
    get_hardest_topic,
    get_individual_student_report,
    create_student,
    get_student,
    get_all_students,
    get_performance_history,
    save_question,
    get_random_question,
    get_questions_by_topic,
)
from schemas import (
    SubmitAnswer,
    SubmitAnswerResponse,
    StudentProgress,
    GeneratedQuestion,
    StoredQuestion,
    StruggleReport,
    TopicDifficulty,
    IndividualReport,
    CreateStudent,
    Student,
    PerformanceHistory,
)
import logging

# This makes all the log messages show up in the terminal with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="AI-Powered Adaptive Learning System")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Create database tables when the app starts up (if they don't exist)."""
    from database import init_db
    await init_db()
    print("Database tables ready.")


# ── Root / Health ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Shows available endpoints so you know what to test."""
    return {
        "message": "AI-Powered Adaptive Learning System",
        "docs": "/docs",
        "endpoints": {
            "POST /students": "Create a student",
            "GET /students": "List all students",
            "GET /students/{id}": "Get one student",
            "GET /students/{id}/performance-history": "Student performance cache",
            "GET /generate-question?topic=math&difficulty=medium": "Generate AI question",
            "GET /get-question?topic=math&difficulty=medium": "Get stored or new question",
            "GET /adaptive-question?student_id=1&topic=math": "Adaptive question",
            "GET /questions/{topic}": "List stored questions",
            "POST /submit-answer": "Submit answer + get AI feedback",
            "GET /student-progress/{id}": "Progress by topic",
            "GET /analytics/struggling-students": "Students below threshold",
            "GET /analytics/hardest-topics": "Topics with low accuracy",
            "GET /analytics/student-report/{id}": "Full report for one student",
            "GET /health": "Check API and database status",
        }
    }


@app.get("/health")
async def health_check():
    """Checks that the API is running and the database is reachable."""
    status = {"status": "API is running"}
    try:
        conn = await get_connection()
        await conn.fetchval("SELECT 1")
        await conn.close()
        status["db"] = "connected"
    except Exception as e:
        status["db"] = f"error: {str(e)}"
    return status


# ── Student management ────────────────────────────────────────────────────

@app.post("/students", response_model=Student)
async def create_new_student(data: CreateStudent):
    """Creates a new student. Returns 400 if the email is already taken."""
    conn = None
    try:
        conn = await get_connection()
        student_id = await create_student(conn, data.name, data.email)
        student = await get_student(conn, student_id)
    except asyncpg.exceptions.UniqueViolationError:
        # This specific error means the email column has a duplicate
        raise HTTPException(status_code=400, detail="A student with that email already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

    return Student(**student)


@app.get("/students", response_model=list[Student])
async def list_students():
    """Returns all students in the system."""
    conn = None
    try:
        conn = await get_connection()
        students = await get_all_students(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

    return [Student(**s) for s in students]


@app.get("/students/{student_id}", response_model=Student)
async def get_student_by_id(student_id: int):
    """Gets one student by their ID. Returns 404 if not found."""
    conn = None
    try:
        conn = await get_connection()
        student = await get_student(conn, student_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    return Student(**student)


@app.get("/students/{student_id}/performance-history", response_model=list[PerformanceHistory])
async def student_performance_history(student_id: int):
    """Gets the cached performance history for a student across all topics."""
    conn = None
    try:
        conn = await get_connection()
        history = await get_performance_history(conn, student_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

    return [PerformanceHistory(**h) for h in history]


# ── Quiz submission ────────────────────────────────────────────────────────

@app.post("/submit-answer", response_model=SubmitAnswerResponse)
async def submit_answer(data: SubmitAnswer):
    """
    Main quiz endpoint - the core flow of the app.

    Steps:
    1. Check if the answer is correct (simple string comparison)
    2. Call Groq AI to generate personalised feedback
    3. Save the result + feedback to the database
    4. Return the result to the student

    If Groq fails (timeout, wrong key, etc.) we still save the result
    to the database with a basic fallback message. This is important -
    we don't want to lose quiz data just because the AI is slow.
    """
    logger = logging.getLogger(__name__)

    # Step 1: Check correctness (case-insensitive, ignore extra spaces)
    is_correct = (
        data.student_answer.strip().lower() == data.correct_answer.strip().lower()
    )
    logger.info(
        f"Student {data.student_id} answered '{data.topic}': "
        f"{'CORRECT' if is_correct else 'INCORRECT'}"
    )

    # Step 2: Get AI feedback
    # Note: get_ai_feedback already handles Groq errors and returns a
    # fallback message, so this won't raise an exception
    try:
        feedback = await get_ai_feedback(
            data.topic,
            data.question,
            data.student_answer,
            data.correct_answer,
            is_correct,
        )
        logger.info("AI feedback received")
    except Exception as e:
        # This shouldn't happen since get_ai_feedback catches its own errors,
        # but just in case something unexpected goes wrong
        logger.error(f"Unexpected feedback error: {e}")
        if is_correct:
            feedback = "Correct! Well done."
        else:
            feedback = f"Incorrect. The correct answer is: {data.correct_answer}"

    # Step 3: Save to database
    conn = None
    try:
        conn = await get_connection()
        await save_quiz_result(conn, data, is_correct, feedback)
        logger.info(f"Saved quiz result for student {data.student_id}")
    except asyncpg.exceptions.ForeignKeyViolationError:
        # This happens if student_id doesn't exist in the students table
        raise HTTPException(
            status_code=400,
            detail=f"Student with ID {data.student_id} does not exist. Create the student first."
        )
    except asyncpg.exceptions.UniqueViolationError as e:
        logger.error(f"Database unique violation: {e}")
        raise HTTPException(status_code=400, detail="Duplicate entry — this result may already exist")
    except Exception as e:
        logger.error(f"Failed to save quiz result: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save quiz result: {str(e)}"
        )
    finally:
        if conn:
            await conn.close()

    # Step 4: Return response to the student
    return SubmitAnswerResponse(
        student_id=data.student_id,
        is_correct=is_correct,
        feedback=feedback,
        message="Answer saved successfully",
    )


# ── Progress ───────────────────────────────────────────────────────────────

@app.get("/student-progress/{student_id}", response_model=list[StudentProgress])
async def student_progress(student_id: int):
    """
    Shows a student's performance grouped by topic.
    Returns total attempts, correct attempts, and accuracy per topic.
    """
    conn = None
    try:
        conn = await get_connection()
        progress_data = await get_student_progress(conn, student_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

    if not progress_data:
        raise HTTPException(
            status_code=404,
            detail="No quiz results found for this student. Submit some answers first!"
        )

    return [StudentProgress(**row) for row in progress_data]


# ── Question endpoints ─────────────────────────────────────────────────────

@app.get("/generate-question", response_model=GeneratedQuestion)
async def generate_question(topic: str, difficulty: str = "medium"):
    """
    Generates a brand new quiz question using Groq AI.

    The question is automatically saved to the database so it can be
    reused later (this saves Groq API calls).

    Query params:
    - topic: e.g. 'fractions', 'world history', 'python basics'
    - difficulty: 'easy', 'medium', or 'hard' (default: medium)
    """
    try:
        question_obj = await generate_quiz_question(topic, difficulty)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")

    # Save the question to the database for future reuse
    conn = None
    try:
        conn = await get_connection()
        await save_question(
            conn,
            topic,
            difficulty,
            question_obj["question"],
            question_obj["correct_answer"],
            question_obj["choices"],
        )
        print(f"Saved new question for topic '{topic}' to database")
    except Exception as e:
        # Don't fail the whole request if saving fails - still return the question
        print(f"Warning: could not save question to database: {e}")
    finally:
        if conn:
            await conn.close()

    return question_obj


@app.get("/get-question", response_model=GeneratedQuestion)
async def get_question(topic: str, difficulty: str = "medium", use_stored: bool = True):
    """
    Gets a quiz question, trying stored questions first.

    If use_stored=True (default), checks the database first to avoid
    unnecessary Groq API calls. Falls back to AI generation if nothing
    is stored yet.

    Query params:
    - topic, difficulty: same as /generate-question
    - use_stored: set to false to always generate a fresh question
    """
    if use_stored:
        conn = None
        try:
            conn = await get_connection()
            stored = await get_random_question(conn, topic, difficulty)
            if stored:
                return GeneratedQuestion(
                    question=stored["question"],
                    correct_answer=stored["correct_answer"],
                    choices=stored["choices"],
                )
        except Exception as e:
            print(f"Error fetching stored question: {e}")
        finally:
            if conn:
                await conn.close()

    # No stored question found (or use_stored=False), generate a new one
    return await generate_question(topic, difficulty)


@app.get("/adaptive-question", response_model=GeneratedQuestion)
async def get_adaptive_question(student_id: int, topic: str):
    """
    Returns a question with difficulty adapted to the student's history.

    The algorithm:
    - accuracy > 80%  → hard question (student is doing well, challenge them)
    - accuracy 60-80% → medium question (on track, keep steady)
    - accuracy < 60%  → easy question (struggling, let them rebuild confidence)
    - no history yet  → medium question (safe default for new topics)

    Query params:
    - student_id: the student's ID
    - topic: the subject area
    """
    conn = None
    difficulty = "medium"  # default if we can't determine from history

    try:
        conn = await get_connection()

        # Check if student has a performance history for this topic
        history = await get_performance_history(conn, student_id)
        topic_history = next((h for h in history if h["topic"] == topic), None)

        if topic_history:
            accuracy = topic_history["average_accuracy"]
            if accuracy > 80:
                difficulty = "hard"
            elif accuracy >= 60:
                difficulty = "medium"
            else:
                difficulty = "easy"
            print(f"Student {student_id} has {accuracy}% on '{topic}' → using {difficulty}")
        else:
            print(f"No history for student {student_id} on '{topic}' → defaulting to medium")

        # Try to serve a stored question at the right difficulty
        stored = await get_random_question(conn, topic, difficulty)
        if stored:
            return GeneratedQuestion(
                question=stored["question"],
                correct_answer=stored["correct_answer"],
                choices=stored["choices"],
            )

    except Exception as e:
        print(f"Error in adaptive question logic: {e}")
        difficulty = "medium"
    finally:
        if conn:
            await conn.close()

    # No stored question - generate one with the right difficulty
    return await generate_question(topic, difficulty)


@app.get("/questions/{topic}", response_model=list[StoredQuestion])
async def list_questions_by_topic(topic: str, limit: int = 10):
    """Lists questions stored in the database for a given topic."""
    conn = None
    try:
        conn = await get_connection()
        questions = await get_questions_by_topic(conn, topic, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

    return [StoredQuestion(**q) for q in questions]


# ── Analytics (teacher dashboard) ─────────────────────────────────────────

@app.get("/analytics/struggling-students", response_model=list[StruggleReport])
async def struggling_students(threshold: float = 70.0):
    """
    Returns students whose overall accuracy is below the threshold.

    Uses SQL HAVING to filter on the aggregated accuracy value.
    WHERE can't do this because the accuracy doesn't exist until after
    the GROUP BY runs.

    Query params:
    - threshold: e.g. 70.0 means "below 70% accuracy" (default: 70.0)
    """
    conn = None
    try:
        conn = await get_connection()
        students = await get_struggling_students(conn, threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

    return [StruggleReport(**row) for row in students]


@app.get("/analytics/hardest-topics", response_model=list[TopicDifficulty])
async def hardest_topics():
    """
    Returns all topics ranked from hardest to easiest (by average accuracy).
    Useful for teachers to see which subjects need more curriculum support.
    """
    conn = None
    try:
        conn = await get_connection()
        topics = await get_hardest_topic(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

    if not topics:
        raise HTTPException(status_code=404, detail="No quiz data yet. Submit some answers first!")

    return [TopicDifficulty(**row) for row in topics]


@app.get("/analytics/student-report/{student_id}", response_model=list[IndividualReport])
async def student_report(student_id: int):
    """
    Gives a detailed breakdown of one student's performance by topic.
    Topics are sorted weakest to strongest so problem areas are obvious.

    Path param:
    - student_id: the student's ID
    """
    conn = None
    try:
        conn = await get_connection()
        report = await get_individual_student_report(conn, student_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"No quiz results found for student {student_id}"
        )

    return [IndividualReport(**row) for row in report]
