from bs4 import BeautifulSoup
import re
import os
import pandas as pd
from datetime import datetime

# 1. Read the file content
# def TextCategorize():
#     dir_path = get_Html_dir_path()


#     for filename in os.listdir(dir_path):
#             # Construct the full path to the file
#         file_path = os.path.join(dir_path, filename)
            
#             # Check if it is actually a file (not a subdirectory)
#         if os.path.isfile(file_path):
#             print(f"Found File: {filename}")

#         with open(file_path, 'r', encoding='utf-8') as f:
#             html_content = f.read()

#         # 2. Parse the HTML and get the text
#         soup = BeautifulSoup(html_content, 'html.parser')
#         plain_text = soup.get_text(separator=' ', strip=True)

#         # plain_text is what you will send to the API
#         print(plain_text[:50])


def get_Html_dir_path( path = os.path.abspath(__file__)):
    # print(os.path.abspath(__file__))
    # Get the current directory of the script
    i = 1
    print(path)
    iteration = -1
    while i : 
        iteration = iteration + 1
        print(iteration)
        current_dir = os.path.dirname(path)
        print(current_dir)
        last_folder = os.path.basename(os.path.normpath(current_dir))
        print(last_folder)
        print(f"Last folder in '{current_dir}': {last_folder}")

        if last_folder == 'Finance-Management-using-AI':
            i = 0
        else :
            path = current_dir

    print(f'current directory at the end = {current_dir}')
    dir_path = os.path.join(current_dir, 'data/html')

    # print(Html_file_path)
    return dir_path

def create_csv():
    df = pd.DataFrame(columns=['Date', 'Category', 'Amount'])
    df.to_csv('../data/history.csv',index = False )

def join_df(row_tuple):
    file_path = '../data/history.csv'
    # create DataFrame for the new row
    new_row = pd.DataFrame([row_tuple])
    # append without reading the file
    new_row.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

def format_response(): 

    input_string = "Amount: 20.00 (USD) ,Purchase_Category: ATM"
    text = """---------- Forwarded message ---------
            From: Chase < no.reply.alerts@chase.com >
            Date: Fri, 31 Oct, 2025, 8:49 pm
            Subject: Here's your atm receipt
            To: < vedantranade2612@gmail.com >
            Details from your ATM visit on 10/31/2025 08:49 PM ."""
    
    match1 = re.match(r"Amount: ([\d.]+) \((\w+)\) ,Purchase_Category: (\w+)", input_string)
    match2 = re.search(r'\b\d{1,2}\s+\w{3,9},\s*\d{4}\b', text)

    if match1:
        # Create the tuple from the extracted data
        data_tuple_1 = (float(match1.group(1)), match1.group(3))
        print(data_tuple_1)
        # # Use .loc to append the tuple as a new row
        # df.loc[len(df)] = new_data_tuple
        
        # print(df)
    else:
        print("String format did not match the expected pattern.")

    if match2:
        # Create the tuple from the extracted data
        data_tuple_2 = (datetime.strptime(match2.group(), "%d %b, %Y"))
        print(data_tuple_2)
        # # Use .loc to append the tuple as a new row
        # df.loc[len(df)] = new_data_tuple
        
        # print(df)
    else:
        print("String format did not match the expected pattern.")

    join_df((data_tuple_2,) + data_tuple_1)
    print("Joined")


if __name__ == '__main__':
    # create_csv()
    format_response()