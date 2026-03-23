import asyncpg
from fastapi import FastAPI, HTTPException
from database import get_connection
from crud import (
    save_quiz_result, get_ai_feedback, get_student_progress,
    generate_quiz_question, get_struggling_students, get_hardest_topic,
    get_individual_student_report, create_student, get_student,
    get_all_students, get_performance_history, save_question,
    get_random_question, get_questions_by_topic,
)
from schemas import (
    SubmitAnswer, SubmitAnswerResponse, StudentProgress, GeneratedQuestion,
    StoredQuestion, StruggleReport, TopicDifficulty, IndividualReport,
    CreateStudent, Student, PerformanceHistory,
)
import logging

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


@app.on_event("startup")
async def startup_event():
    """Initialises database tables on startup."""
    from database import init_db
    await init_db()
    print("Database tables ready.")


@app.get("/")
def root():
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
    """Checks API and database connectivity."""
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
        raise HTTPException(status_code=400, detail="A student with that email already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()
    return Student(**student)


@app.get("/students", response_model=list[Student])
async def list_students():
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
    """Returns 404 if the student is not found."""
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
    """Checks the answer, gets AI feedback, saves the result, and returns a response."""
    logger = logging.getLogger(__name__)

    is_correct = data.student_answer.strip().lower() == data.correct_answer.strip().lower()
    logger.info(f"Student {data.student_id} answered '{data.topic}': {'CORRECT' if is_correct else 'INCORRECT'}")

    try:
        feedback = await get_ai_feedback(
            data.topic, data.question, data.student_answer, data.correct_answer, is_correct,
        )
        logger.info("AI feedback received")
    except Exception as e:
        logger.error(f"Unexpected feedback error: {e}")
        feedback = "Correct! Well done." if is_correct else f"Incorrect. The correct answer is: {data.correct_answer}"

    conn = None
    try:
        conn = await get_connection()
        await save_quiz_result(conn, data, is_correct, feedback)
        logger.info(f"Saved quiz result for student {data.student_id}")
    except asyncpg.exceptions.ForeignKeyViolationError:
        raise HTTPException(
            status_code=400,
            detail=f"Student with ID {data.student_id} does not exist. Create the student first."
        )
    except asyncpg.exceptions.UniqueViolationError as e:
        logger.error(f"Database unique violation: {e}")
        raise HTTPException(status_code=400, detail="Duplicate entry — this result may already exist")
    except Exception as e:
        logger.error(f"Failed to save quiz result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save quiz result: {str(e)}")
    finally:
        if conn:
            await conn.close()

    return SubmitAnswerResponse(
        student_id=data.student_id,
        is_correct=is_correct,
        feedback=feedback,
        message="Answer saved successfully",
    )


# ── Progress ───────────────────────────────────────────────────────────────

@app.get("/student-progress/{student_id}", response_model=list[StudentProgress])
async def student_progress(student_id: int):
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
        raise HTTPException(status_code=404, detail="No quiz results found for this student.")
    return [StudentProgress(**row) for row in progress_data]


# ── Question endpoints ─────────────────────────────────────────────────────

@app.get("/generate-question", response_model=GeneratedQuestion)
async def generate_question(topic: str, difficulty: str = "medium"):
    """Generates a new question via Groq and saves it to the database."""
    try:
        question_obj = await generate_quiz_question(topic, difficulty)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")

    conn = None
    try:
        conn = await get_connection()
        await save_question(conn, topic, difficulty, question_obj["question"],
                            question_obj["correct_answer"], question_obj["choices"])
        print(f"Saved new question for topic '{topic}' to database")
    except Exception as e:
        print(f"Warning: could not save question to database: {e}")
    finally:
        if conn:
            await conn.close()

    return question_obj


@app.get("/get-question", response_model=GeneratedQuestion)
async def get_question(topic: str, difficulty: str = "medium", use_stored: bool = True):
    """Returns a stored question if available, otherwise generates a new one."""
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

    return await generate_question(topic, difficulty)


@app.get("/adaptive-question", response_model=GeneratedQuestion)
async def get_adaptive_question(student_id: int, topic: str):
    """
    Returns a question with difficulty adapted to the student's history.
    - accuracy > 80%  → hard
    - accuracy 60-80% → medium
    - accuracy < 60%  → easy
    - no history      → medium (default)
    """
    conn = None
    difficulty = "medium"

    try:
        conn = await get_connection()
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

    return await generate_question(topic, difficulty)


@app.get("/questions/{topic}", response_model=list[StoredQuestion])
async def list_questions_by_topic(topic: str, limit: int = 10):
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


# ── Analytics ──────────────────────────────────────────────────────────────

@app.get("/analytics/struggling-students", response_model=list[StruggleReport])
async def struggling_students(threshold: float = 70.0):
    """Returns students whose overall accuracy is below the threshold."""
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
    """Returns topics ranked from hardest to easiest by average accuracy."""
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
        raise HTTPException(status_code=404, detail="No quiz data yet.")
    return [TopicDifficulty(**row) for row in topics]


@app.get("/analytics/student-report/{student_id}", response_model=list[IndividualReport])
async def student_report(student_id: int):
    """Returns a student's performance breakdown by topic, weakest first."""
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
        raise HTTPException(status_code=404, detail=f"No quiz results found for student {student_id}")
    return [IndividualReport(**row) for row in report]
