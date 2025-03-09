import psycopg2

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname="your_database_name",     # Replace with your database name
    user="your_username",       # Replace with your username
    password="your_password",  # Replace with your password
    host="localhost",          # Replace with your host, e.g., localhost or IP
    port="5432"                # Default PostgreSQL port
)

# Create a cursor object to interact with the database
cur = conn.cursor()

# Create the student table
cur.execute("""
    CREATE TABLE IF NOT EXISTS student (
        student_id SERIAL PRIMARY KEY,
        first_name VARCHAR(50),
        last_name VARCHAR(50),
        age INT,
        email VARCHAR(100),
        enrolled_date DATE
    );
""")

# Insert sample student records into the table
cur.execute("""
    INSERT INTO student (first_name, last_name, age, email, enrolled_date)
    VALUES 
    ('John', 'Doe', 20, 'john.doe@example.com', '2023-01-15'),
    ('Jane', 'Smith', 22, 'jane.smith@example.com', '2022-09-10'),
    ('Mark', 'Johnson', 19, 'mark.johnson@example.com', '2023-02-01'),
    ('Emily', 'Davis', 21, 'emily.davis@example.com', '2021-05-05');
""")

# Commit the changes
conn.commit()

# Fetch and print all the records from the student table
cur.execute("SELECT * FROM student;")
students = cur.fetchall()

for student in students:
    print(student)

# Close the cursor and connection
cur.close()
conn.close()
