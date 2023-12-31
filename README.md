## Libraries

- NumPy, Pandas, scikit-learn, TensorFlow, PyTorch

## Instructions

- Open `chatbot/chatbot.ipynb`, click `Open in Colab`
- Open `data/`, download `courses.csv`, `sentences.csv`
- Open `parameters/`, download `intent.pth`
- Upload `courses.csv`, `sentences.csv` and `intent.pth`
- Choose `CPU` runtime and run the notebook

## Manual
1. **Commands**:

- `bye` - End conversation
- `help` - Display manual
- `list courses` - Show courses list

2. **Conversation Flow**:

a. Enroll a course:
- Example: "I want to enroll in 6.4100 for Fall 2024 to learn AI."

b. Search for courses:
- Example: "What are the undergraduate physics classes offered in Spring 2024?"
- Note: only support course name in `list courses`

c. Show detailed course information:
- Example: "Can you provide detail information of course 6.4100?"

##  References

- MIT Course Catalog: https://catalog.mit.edu/
- Stanford CS230 Autumn 2018 - Lecture 10: https://www.youtube.com/watch?v=IFLstgCNOA4&t=203s