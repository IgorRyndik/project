import sqlite3
import bcrypt

def hash_password(password):
    # Generate a salt
    salt = bcrypt.gensalt()
    
    # Hash the password using the salt
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    return hashed_password

def add_user(user_id, username, password):
    # Connect to the SQLite database (or create a new one)
    connection = sqlite3.connect('users.db')

    # Create a cursor object to execute SQL commands
    cursor = connection.cursor()

    # Insert data into the table
    cursor.execute('INSERT INTO users (user_id, username, hashed_password) VALUES (?, ?, ?)', 
                   (user_id, username,hash_password(password)))

    # Commit the changes
    connection.commit()

    # Close the connection
    connection.close()

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password)

def check_user_password(user_id: int, password):
    # Connect to the SQLite database (or create a new one)
    connection = sqlite3.connect('users.db')

    # Create a cursor object to execute SQL commands
    cursor = connection.cursor()

    # Query data from the table
    cursor.execute('SELECT hashed_password FROM users WHERE user_id = ?', (user_id, ))

    row = cursor.fetchone()

    hashed_password, = row

    print(hashed_password)

    # Close the connection
    connection.close()

    if hashed_password:
        return verify_password(password, hashed_password)
    else:
        return false

if __name__ == '__main__':
    # Connect to the SQLite database (or create a new one)
    connection = sqlite3.connect('users.db')

    # Create a cursor object to execute SQL commands
    cursor = connection.cursor()

    # Define a table schema and create the table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            username TEXT,
            hashed_password TEXT
        )
    ''')

    # Commit the changes
    connection.commit()

    # Close the connection
    connection.close()

    #add_user(45, 'German', '0123')
    #add_user(6, 'Kino', '9876')
    print(check_user_password(45, '0123'))
