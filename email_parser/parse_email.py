# from imap_tools import MailBox

# password = 'cndyupggwphmicvu'
# username = 'ai.39.vedant.ranade@gmail.com'

# with MailBox("imap.gmail.com").login(username=username, password= password,initial_folder = 'INBOX') as mb:
#     print(mb.folder.list())
#     for email in mb.fetch(limit= 1,reverse=True, mark_seen = False):
#         print(email.subject, email.date, email.html)

# # print('HII')

from imap_tools import MailBox, AND
import re

# Define your IMAP server and credentials
IMAP_USER = 'ai.39.vedant.ranade@gmail.com'
IMAP_PASSWORD = 'cndyupggwphmicvu'
IMAP_SERVER = 'imap.gmail.com'

# KEYWORDS = ['Chase Bank', 'tranasaction']
KEYWORDS = [
    "chase bank", 
    "transaction", 
    "purchase", 
    "debited", 
    "credited", 
    "payment", 

    "withdrawal",
    "deposit", 
    "bill", 
    "invoice", 
    "due", 
    "autopay",
    "successfully paid",
    "received payment",
    "utility bill",
    "rent",
    "order confirmation",
    "receipt",
    "thank you for your payment",
    "your payment was received"
]

def clean_subject(subject):

    ### fOR TESING ONLY
    print(subject)

    ###################333

    # Extract first 10 words and make filesystem-safe
    words = re.findall(r'\w+', subject)
    safe_subject = '_'.join(words[1:2])
    return safe_subject

def matches_keywords(msg):
    check_text = f"{msg.subject} {msg.text}".lower()
    
    # fOR TESTING ONLY
    matches = [k for k in KEYWORDS if k in check_text]
    print(matches)

    ##################################333
    return any(k in check_text for k in KEYWORDS)

# Connect to mailbox and search
with MailBox(IMAP_SERVER).login(IMAP_USER, IMAP_PASSWORD, 'INBOX') as mailbox:
    # Fetch all emails (could also add date filters in AND if desired)
    for msg in mailbox.fetch(limit = 5 , reverse = True):
        if matches_keywords(msg):
            # Get HTML content
            html_content = msg.html if msg.html else msg.text  # fallback to plain text if no html
            # Save HTML for PDF conversion later

            date_str = msg.date.strftime('%d%m%y')
            # Subject: first 10 words, cleaned
            subject_str = clean_subject(msg.subject)
            # Final filename
            filename = f"../data/html/{date_str}_{subject_str}.html"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)

            # Later, you can convert these .html files to PDF and store/upload as needed
