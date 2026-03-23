# AI-Powered Adaptive Learning System

A FastAPI-based learning platform that adapts to each student using AI feedback and performance analytics.

## Features

- **Adaptive Question Generation**: Questions adjust difficulty based on student performance
- **Real-time AI Feedback**: Instant feedback using Groq AI for wrong answers
- **Performance Analytics**: Teacher dashboards for identifying struggling students and topics
- **Question Bank**: Stores generated questions for reuse and efficiency
- **Student Management**: Create and track student progress across topics

## Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: Aiven PostgreSQL (connect via DBeaver)
- **AI**: Groq API (llama3-8b-8192)
- **Server**: Uvicorn

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project folder:

```
DATABASE_URL=postgresql://user:password@host:port/database
GROQ_API_KEY=your_groq_api_key
```

### 3. Run the application

```bash
uvicorn routes:app --reload
```

### 4. Open the interactive docs

Go to: http://localhost:8000/docs

From there you can test every endpoint directly in the browser.

---

## Testing the App (Step by Step)

### Step 1: Create a student

```
POST /students
Body: {"name": "Jane Doe", "email": "jane@example.com"}
```

### Step 2: Generate a question

```
GET /generate-question?topic=math&difficulty=easy
```

This calls Groq to create a question **and saves it to your database**. 

### Step 3: Submit an answer

```
POST /submit-answer
Body:
{
  "student_id": 1,
  "topic": "math",
  "question": "What is 5 + 3?",
  "student_answer": "8",
  "correct_answer": "8",
  "time_taken": 10
}
```

Check database — a new row should appear in `quiz_results` and `performance_history`.

### Step 4: View student progress

```
GET /student-progress/1
```

---

## API Endpoints

### Student Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/students` | Create a new student |
| GET | `/students` | List all students |
| GET | `/students/{id}` | Get one student |
| GET | `/students/{id}/performance-history` | Performance cache by topic |

### Quiz
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/submit-answer` | Submit answer + get AI feedback |
| GET | `/student-progress/{id}` | Progress grouped by topic |

### Questions
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/generate-question` | Generate new AI question (saves to DB) |
| GET | `/get-question` | Prefer stored, fall back to AI |
| GET | `/adaptive-question` | Difficulty adapts to student history |
| GET | `/questions/{topic}` | List stored questions by topic |

### Analytics (Teacher Dashboard)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/analytics/struggling-students?threshold=70` | Students below accuracy threshold |
| GET | `/analytics/hardest-topics` | Topics with lowest average accuracy |
| GET | `/analytics/student-report/{id}` | Full breakdown for one student |

---

## Database Tables

| Table | Purpose |
|-------|---------|
| `students` | Student profiles (name, email) |
| `questions` | Stored AI-generated questions |
| `quiz_results` | Every individual quiz attempt |
| `performance_history` | Aggregated accuracy per student per topic |

Tables are created automatically when the app starts. You can view them in DBeaver by connecting with your `DATABASE_URL` credentials.

---

## Adaptive Difficulty Logic

When a student requests an adaptive question the system checks their history:

| Accuracy | Difficulty given |
|----------|-----------------|
| > 80% | Hard |
| 60–80% | Medium |
| < 60% | Easy |
| No history yet | Medium (default) |

---
