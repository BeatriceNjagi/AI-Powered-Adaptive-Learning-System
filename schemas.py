from pydantic import BaseModel
from datetime import datetime

class SubmitAnswer(BaseModel):
    student_id: int
    topic: str
    question: str
    student_answer: str
    correct_answer: str
    time_taken: int | None = None  # optional time in seconds

class SubmitAnswerResponse(BaseModel):
    student_id: int
    is_correct: bool
    feedback: str
    message: str

class StudentProgress(BaseModel):
    topic: str
    total_attempts: int
    correct_attempts: int
    accuracy_percentage: float


class GeneratedQuestion(BaseModel):
    question: str
    correct_answer: str
    choices: list[str] | None = None  # optional, for multiple-choice


class StoredQuestion(BaseModel):
    id: int
    topic: str
    difficulty: str
    question: str
    correct_answer: str
    choices: list[str] | None = None
    created_at: datetime


class StruggleReport(BaseModel):
    student_id: int
    total_attempts: int
    correct_attempts: int
    average_accuracy: float


class TopicDifficulty(BaseModel):
    topic: str
    total_attempts: int
    correct_attempts: int
    average_accuracy: float


class IndividualReport(BaseModel):
    student_id: int
    topic: str
    total_attempts: int
    correct_attempts: int
    accuracy_percentage: float


# Student management schemas
class CreateStudent(BaseModel):
    name: str
    email: str

class Student(BaseModel):
    id: int
    name: str
    email: str
    joined_at: datetime

class PerformanceHistory(BaseModel):
    student_id: int
    topic: str
    total_attempts: int
    correct_attempts: int
    average_accuracy: float
    last_updated: datetime