import os
import yaml

def Config_API( path = os.path.abspath(__file__)):
    # print(os.path.abspath(__file__))
    # Get the current directory of the script
    i = 1
    while i : 
        current_dir = os.path.dirname(path)
        last_folder = os.path.basename(os.path.normpath(current_dir))
        print(f"Last folder in '{current_dir}': {last_folder}")

        if last_folder == 'Expense-Categorization':
            i = 0
        else :
            path = current_dir

    # print(current_dir)
    # Construct the path to the parent directory
    # parent_dir = os.path.join(current_dir, '..')
    # print(parent_dir)

    # Construct the full path to secrets.yaml
    secrets_file_path = os.path.join(current_dir, 'secrets.yaml')
    print(secrets_file_path)

    try:
        with open(secrets_file_path, 'r') as file:
            secrets = yaml.safe_load(file)
        print("Secrets loaded successfully:")
        Gemini_Key = secrets['Gemini_API_key']
        return Gemini_Key

    except FileNotFoundError:
        print(f"Error: secrets.yaml not found at {secrets_file_path}")
    except yaml.YAMLError as e:
        print(f"Error parsing secrets.yaml: {e}")



if __name__ == '__main__':
    key = Config_API()
    print(key)
# # Example path ending with a folder name
# path1 = "E:\FAI_Project\Finance-Management-using-AI\email_category"
# last_folder1 = os.path.basename(os.path.normpath(path1))
# print(f"Last folder in '{path1}': {last_folder1}")

# # Example path ending with a trailing slash
# path2 = os.path.abspath(__file__)
# last_folder2 = os.path.basename(os.path.normpath(path2))
# print(f"Last folder in '{path2}': {last_folder2}")