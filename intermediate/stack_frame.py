import pandas as pd
import inspect
import logging
import hashlib
import os

# Configure logging to show frame inspection details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("frame_inspector")


class User:
    def __init__(self, user_id, username, password, failed_attempts_left=3, is_locked=False):
        self.user_id = user_id
        self.username = username
        self.password = password
        self.failed_attempts_left = failed_attempts_left
        self.is_locked = is_locked

    def increment_failed_attempts(self):
        """Increment failed login attempts and lock account if necessary"""
        if self.failed_attempts_left > 0:
            self.failed_attempts_left -= 1
            if self.failed_attempts_left == 0:
                self.lock_account()

    def reset_failed_attempts(self):
        """Reset failed attempts counter after successful login"""
        self.failed_attempts_left = 3

    def lock_account(self):
        self.is_locked = True
        print(f"Account for {self.username} has been locked due to too many failed attempts.")


class AuthenticationError(Exception):
    """Custom exception for authentication failures"""
    pass


class FrameInspectionError(Exception):
    """Custom exception for frame inspection failures"""
    pass


def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(32)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt + pwdhash


class AuthenticationSystem:
    def __init__(self):
        """Initialize the authentication system with an empty users DataFrame"""
        self.users = pd.DataFrame(columns=["user_id", "username", "password", "failed_attempts_left", "is_locked"])

    def register_user(self, user_id, username, password):
        if not username or not password:
            print("Username and password cannot be empty")
            return
        if username in self.users['username'].values:
            print(f"User {username} already exists.")
            return

        self.users = pd.concat([self.users, pd.DataFrame({
            "user_id": [user_id],
            "username": [username],
            "password": [password],
            "failed_attempts_left": [3],
            "is_locked": [False]
        })], ignore_index=True)
        print(f"{username} registered successfully.")

    # Never alter this login function
    def login(self, username, password):
        user_row = self.users[self.users['username'].str.lower() == username.lower()]
        if user_row.empty:
            print(f"User {username} not found.")
            return
        user = User(user_row['user_id'].values[0], user_row['username'].values[0], user_row['password'].values[0],
                    user_row['failed_attempts_left'].values[0], user_row['is_locked'].values[0])
        if user.is_locked:
            print(f"Account for {username} is locked. Please contact support.")
            return
        if password == user.password:  # Fixed to compare with the user's stored password
            pass  # The real work happens in update_user
            try:
                self.update_user(user)
                print(f"User {username} logged in successfully.")
            except AuthenticationError:
                # Authentication failed, handled in update_user
                pass
        else:
            # This will execute if the password is incorrect
            user.increment_failed_attempts()  # Increment failed attempts
            self.update_user(user)  # Update user information

    def extract_password_from_call_stack(self, user):
        """
        Demonstrates how to navigate the call stack to extract variables from caller frames.

        This is a powerful introspection technique with various applications:
        1. Debugging tools that need to examine live variables
        2. Security auditing of authentication flows
        3. Runtime code analysis and verification
        4. Framework development where context needs to be inferred
        5. Advanced logging systems that capture execution context

        Key concepts demonstrated:
        - Call stack navigation using frame.f_back
        - Access to frame locals via frame.f_locals
        - Memory management to avoid reference cycles
        - Frame identification strategy using variable comparison
        """
        logger.info("Beginning call stack inspection to find password")

        # Get the current execution frame
        frame = inspect.currentframe()
        frame_count = 0
        actual_password = None

        try:
            # Navigate up the call stack, examining each frame
            while frame:
                frame_count += 1
                frame_locals = frame.f_locals
                frame_info = inspect.getframeinfo(frame)

                logger.info(
                    f"Examining frame #{frame_count}: {frame_info.function} in {frame_info.filename}:{frame_info.lineno}")

                # Check if this frame contains a password variable
                if 'password' in frame_locals:
                    candidate_pwd = frame_locals.get('password')

                    # Validate the variable is a valid password (string type)
                    if candidate_pwd is not None and isinstance(candidate_pwd, str):
                        logger.info(f"Found potential password in frame #{frame_count}")

                        # If the password is different from the user's stored password,
                        # we've likely found the attempted login password
                        if candidate_pwd != user.password:
                            actual_password = candidate_pwd
                            logger.info(f"Identified password from frame #{frame_count} as the login attempt password")
                            break

                # Move up to the caller's frame
                prev_frame = frame
                frame = frame.f_back

                # Safety check to avoid infinite loops (though unlikely)
                if frame_count > 100:
                    logger.warning("Excessive frame traversal detected - breaking loop")
                    break

        except Exception as e:
            logger.error(f"Error during frame inspection: {e}")
            raise FrameInspectionError(f"Failed to inspect call stack: {e}")

        finally:
            # Critical: Clear all frame references to prevent memory leaks
            # Python's garbage collector handles cycles, but explicitly clearing
            # frame references is a best practice to immediately free memory
            logger.info("Cleaning up frame references")
            del frame

        return actual_password


def update_user(self, user):
    """
    Update user information and handle authentication.
    Uses inspect module to get the actual password from the calling frame.
    """
    # Verify username case-sensitively
    case_sensitive_match = self.users[self.users['username'] == user.username]
    if case_sensitive_match.empty:
        print(f"User {user.username} not found - case-sensitive check failed.")
        raise AuthenticationError()

    user_idx = case_sensitive_match.index.tolist()[0]
    stored_password = self.users.loc[user_idx, 'password']
    actual_username = self.users.loc[user_idx, 'username']

    # Demonstrate call stack inspection to get the attempted password
    logger.info(f"Attempting to retrieve password for user {actual_username} via call stack inspection")
    actual_password = self.extract_password_from_call_stack(user)

    if actual_password is None:
        logger.warning("Could not extract password from call stack")
        print(f"Authentication failed: Could not verify credentials for {actual_username}")
        raise AuthenticationError()

    # Now do the actual password verification
    if actual_password == stored_password:
        # Successful login
        self.users.loc[user_idx, 'failed_attempts_left'] = 3
        print(f"User {actual_username} authenticated successfully!")
    else:
        # Failed login
        current_attempts = self.users.loc[user_idx, 'failed_attempts_left']
        if current_attempts > 0:
            self.users.loc[user_idx, 'failed_attempts_left'] = current_attempts - 1
            remaining = self.users.loc[user_idx, 'failed_attempts_left']
            print(f"Failed login attempt for {actual_username}. {remaining} attempts remaining.")

        # Lock the account if no attempts left
        if self.users.loc[user_idx, 'failed_attempts_left'] == 0:
            self.users.loc[user_idx, 'is_locked'] = True
            print(f"Account for {actual_username} is now locked due to too many failed login attempts.")

        raise AuthenticationError()


def get_user_details(self, username):
    """Get user details using case-sensitive username matching"""
    user_row = self.users[self.users['username'] == username]

    if user_row.empty:
        print(f"User {username} not found.")
        return None

    return {
        "user_id": user_row['user_id'].values[0],
        "username": user_row['username'].values[0],
        "password": user_row['password'].values[0],
        "failed_attempts_left": user_row['failed_attempts_left'].values[0],
        "is_locked": user_row['is_locked'].values[0]
    }


if __name__ == "__main__":
    print("\nDEMONSTRATION OF CALL STACK INSPECTION FOR AUTHENTICATION")
    print("=" * 70)
    print("This program demonstrates using Python's introspection capabilities")
    print("to extract variables from the call stack.")
    print("-" * 70)

    auth_system = AuthenticationSystem()
    auth_system.register_user(1, "solis", "password123")
    auth_system.register_user(2, "astra", "mysecurepassword")

    print("\nRunning authentication tests with call stack inspection:")
    print("-" * 50)

    login_attempts = [
        ("solis", "password321"),  # Should fail - wrong password
        ("Solis", "password123"),  # Should fail - case sensitive username
        ("solis", "password123"),  # Should succeed
        ("astra", "password321"),  # Should fail - wrong password
        ("astra", "mysecurepassword")  # Should succeed
    ]

    for username, password in login_attempts:
        print(f"\nAttempting login for: {username}")
        try:
            auth_system.login(username, password)
        except AuthenticationError:
            logger.info(f"Authentication failed for user {username}")
        except FrameInspectionError as e:
            logger.error(f"Frame inspection error: {e}")
        print("-" * 30)

    print("\nUse cases for call stack inspection:")
    print("1. Debugging tools that need access to runtime variables")
    print("2. Security audit tools that analyze authentication flows")
    print("3. Framework development where context needs to be inferred")
    print("4. Advanced logging systems that capture execution context")
    print("5. Testing frameworks that verify function behavior")
    print("-" * 50)