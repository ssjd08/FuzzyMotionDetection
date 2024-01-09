import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email():
    # Set your email and password
    sender_email = "sajadporkhadne08@gmail.com"
    sender_password = "amhmksicmwunkzio"
    # sender_password = "09152536034sa"


    # Set the recipient email address
    recipient_email = "sajadporkhadnemanesh@gmail.com"

    # Create the email message
    subject = "motion detected"
    body = "This is a test email sent from Python."
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    # Connect to the SMTP server (in this case, Gmail's SMTP server)
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        # Start the TLS connection
        server.starttls()

        # Log in to the email account
        server.login(sender_email, sender_password)

        # Send the email
        server.sendmail(sender_email, recipient_email, message.as_string())

    print("Email sent successfully")
