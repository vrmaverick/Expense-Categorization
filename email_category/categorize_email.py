import APIConfig
from google import genai
import os
from preprocessing import * 

def categorize_email(text):
    GEMINI_API_KEY = APIConfig.Config_API()


    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client(api_key = GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=f"""Analyze the following text: {text}.
        Find the transaction amount and categorize the purchase using a single entry from the following list:
        [ATM, Travel, Shopping, Bills, Entertainment].
        Format the output as 'Amount: (USD) ,Purchase_Category: [Category]'.
        If no purchase data is found, return only the number 0."""
    )
    print(response.text)
    format_response(response,text)



def TextExtraction():
    dir_path = get_Html_dir_path()


    for filename in os.listdir(dir_path):
            # Construct the full path to the file
        file_path = os.path.join(dir_path, filename)
            
            # Check if it is actually a file (not a subdirectory)
        if os.path.isfile(file_path):
            print(f"Found File: {filename}")

        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # 2. Parse the HTML and get the text
        soup = BeautifulSoup(html_content, 'html.parser')
        plain_text = soup.get_text(separator=' ', strip=True)

        # plain_text is what you will send to the API
        print(plain_text[:50])
        categorize_email(plain_text)

if __name__ == '__main__':
    TextExtraction()