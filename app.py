import os
import psycopg2
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    # Fix Render postgres:// compatibility
    if DATABASE_URL.startswith("postgres://"):
        db_url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    else:
        db_url = DATABASE_URL

    return psycopg2.connect(db_url)


def create_table():
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            name TEXT,
            age INT,
            course TEXT,
            email TEXT
        )
        """)

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("Database connection error:", e)


# Run table creation when app starts
create_table()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/add_student", methods=["POST"])
def add_student():

    name = request.form["name"]
    age = request.form["age"]
    course = request.form["course"]
    email = request.form["email"]

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO students (name, age, course, email) VALUES (%s,%s,%s,%s)",
        (name, age, course, email),
    )

    conn.commit()
    cur.close()
    conn.close()

    return redirect("/students")


@app.route("/students")
def view_students():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM students")
    students = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("students.html", students=students)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)