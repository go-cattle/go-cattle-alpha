import datetime


def highlight_list_elements(lst):
        highlighted_list = ""
        for item,n in enumerate(lst):
            highlighted_list += f'<span style="background-color: white;">{item+1}). {n} </span>\n'
        return highlighted_list

def save_feedback(name,email,feedback):
    # Generate a unique filename using the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"./feedbacks/feedback_{name}_{email}_{timestamp}.txt"

    # Save the feedback to a file
    with open(filename, 'w') as file:
        file.write(f'name : {name}\nemail : {email}\nfeedback : {feedback}')

    return filename

if __name__ == "__main__":
    print("This is the main module.")