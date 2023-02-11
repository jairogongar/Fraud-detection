import sqlite3

# Connect to the database or create it if it doesn't exist
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# Create the users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username text PRIMARY KEY,
    password text NOT NULL
)
""")
conn.commit()

# Insert a new user into the database
def add_user(username, password):
    try:
        cursor.execute("""
        INSERT INTO users (username, password)
        VALUES (?, ?)
        """, (username, password))
        conn.commit()
        print("User added successfully.")
    except sqlite3.IntegrityError:
        print("Error: User with that username already exists.")

# Check if a username and password match an existing user in the database
def check_login(username, password):
    cursor.execute("""
    SELECT * FROM users
    WHERE username = ? AND password = ?
    """, (username, password))
    return cursor.fetchone() is not None

# Example usage

