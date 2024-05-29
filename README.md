
## GitHub Issue Classification Model

### Purpose
The primary purpose of the GitHub issue classification model is to automatically categorize and prioritize issues posted in a repository. This helps in managing the repository more efficiently by ensuring that issues are appropriately labeled and handled according to their type and urgency.

### Aim
The aim of the model is to improve the workflow of developers and maintainers by reducing the manual effort required to classify and manage issues. By automating the classification process, the model seeks to enhance productivity, streamline issue management, and improve response times to critical issues.

### Objectives
1. **Automate Issue Categorization:** Develop a model that can automatically classify issues into predefined categories such as bug reports, feature requests, documentation updates, questions, etc.
2. **Enhance Accuracy:** Ensure that the model achieves high accuracy in classification to minimize the need for manual corrections and increase trust in the system.
3. **Improve Prioritization:** Integrate priority classification to help identify and highlight urgent and high-impact issues that need immediate attention.
4. **Facilitate Tagging:** Automate the tagging process to help in better issue tracking, searching, and filtering within the repository.
5. **Support Consistency:** Maintain consistent labeling and categorization across issues, reducing the variability introduced by different contributors manually classifying issues.
6. **Reduce Manual Workload:** Significantly decrease the time and effort required by maintainers to triage and manage incoming issues.
7. **Enhance Collaboration:** Improve communication and collaboration among contributors by providing clear and consistent categorization of issues, making it easier to assign tasks and track progress.
8. **Scalability:** Ensure the model can handle a large number of issues, making it suitable for both small and large repositories.
9. **User Feedback Integration:** Implement mechanisms to incorporate feedback from users to continually improve the model's performance and adapt to evolving repository needs.

By achieving these objectives, the GitHub issue classification model can substantially improve the efficiency of project management and contribute to the overall health and productivity of open-source projects and software development teams.

### Methodology
This research project involves a comprehensive exploration of existing literature and state-of-the-art practices. 

Initially, real-time raw text data was meticulously collected online following an extensive research process to ensure the reliability of the data for the project. The link to the data is saved in this repo as link.txt. 

The data underwent a thorough cleaning and preprocessing phase, which included tasks such as eliminating irrelevant characters, addressing typos, handling missing values, and converting text data to lowercase. Stop words like "and," "this," and "the" were removed, and issues related to symbols, emojis, and special characters were addressed. 

Relevant features were then extracted from the processed text data to enhance the model. Exploratory data analysis was conducted to gain insights into the data distribution, with a consideration of its balance. Subsequently, the data was divided into training and testing sets.

Multiple models were trained on the data to identify the most suitable one. The chosen model was then evaluated using various metrics, and the most accurate version was saved. The model was finally tested using unseen data, and conclusions were drawn based on the outcomes of the evaluation.
