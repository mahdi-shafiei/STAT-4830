const markdown = `# STAT 4830: Student Background Survey

## Basic Information
1. Full Name: [Short answer]
2. Penn Email: [Short answer]
3. Major(s)/Minor(s): [Short answer]
4. Expected Graduation Year: [Short answer]
5. Student Status:
   - [ ] Undergraduate
   - [ ] Masters
   - [ ] PhD
   - [ ] Other: [Short answer]

## Programming Background
6. Rate your Python experience:
   - [ ] No experience
   - [ ] Basic (used in 1-2 courses)
   - [ ] Intermediate (multiple courses/projects)
   - [ ] Advanced (significant project work)
   - [ ] Expert (professional experience)

7. Which programming languages have you used? Select all that apply:
   - [ ] Python
   - [ ] Java
   - [ ] C/C++
   - [ ] JavaScript
   - [ ] R
   - [ ] MATLAB
   - [ ] Other: [Short answer]

8. Rate your comfort with:
   * Writing functions and classes:
     (1: Not comfortable → 5: Very comfortable) [Linear scale]
   * Using Git/GitHub:
     (1: Never used → 5: Use regularly) [Linear scale]
   * Debugging code:
     (1: Need significant help → 5: Debug independently) [Linear scale]
   * Linux command line:
     (1: Never used → 5: Daily user) [Linear scale]

9. What development environments do you use? Select all that apply:
   - [ ] VS Code
   - [ ] Jupyter notebooks
   - [ ] Google Colab
   - [ ] Terminal + vim/emacs
   - [ ] PyCharm
   - [ ] Other: [Short answer]

## Mathematics Background
10. When did you last use these linear algebra concepts? Select all used in the past year:
    - [ ] Matrix multiplication
    - [ ] Vector spaces
    - [ ] Eigenvalues/eigenvectors
    - [ ] Matrix factorization
    - [ ] Inner products
    - [ ] None of these

11. Rate your comfort with calculus concepts:
    * Computing gradients:
      (1: Not comfortable → 5: Very comfortable) [Linear scale]
    * Chain rule with vectors/matrices:
      (1: Not comfortable → 5: Very comfortable) [Linear scale]
    * Optimization problems:
      (1: Not comfortable → 5: Very comfortable) [Linear scale]

12. Are you familiar with these terms? Select all that apply:
    - [ ] Jacobian
    - [ ] Hessian
    - [ ] Convexity
    - [ ] Gradient descent
    - [ ] None of these

## Statistical Background
13. Rate your comfort with:
    * Probability distributions:
      (1: Basic understanding → 5: Advanced applications) [Linear scale]
    * Maximum likelihood estimation:
      (1: Never used → 5: Comfortable implementing) [Linear scale]
    * Statistical modeling:
      (1: Limited exposure → 5: Built multiple models) [Linear scale]

## Machine Learning Experience
14. Your ML experience includes: Select all that apply:
    - [ ] Reading about ML concepts
    - [ ] Completing online courses
    - [ ] Course projects
    - [ ] Research projects
    - [ ] Industry experience
    - [ ] No experience

15. Rate your experience with:
    * PyTorch:
      (1: Never used → 5: Significant experience) [Linear scale]
    * Training neural networks:
      (1: Never done → 5: Multiple projects) [Linear scale]
    * Implementing ML algorithms:
      (1: Never done → 5: Comfortable implementing) [Linear scale]

## Computing Resources
16. What GPU access do you have? Select all that apply:
    - [ ] Personal computer with GPU
    - [ ] Cloud GPU resources (AWS, GCP, etc.)
    - [ ] University resources
    - [ ] No access

17. Which AI tools do you use? Select all that apply:
    - [ ] ChatGPT
    - [ ] GitHub Copilot
    - [ ] Claude
    - [ ] None of these

18. Have you installed Cursor IDE?
    - [ ] Yes
    - [ ] No
    - [ ] Not yet, but planning to

## Research and Project Experience
19. Rate your experience with:
    * Reading academic papers:
      (1: Never read → 5: Read regularly) [Linear scale]
    * Implementing algorithms from papers:
      (1: Never tried → 5: Implemented multiple) [Linear scale]
    * Giving technical presentations:
      (1: No experience → 5: Comfortable presenting) [Linear scale]

20. What project areas interest you? Select all that apply:
    - [ ] Training/fine-tuning models
    - [ ] Reproducibility studies
    - [ ] Educational tools
    - [ ] Optimization benchmarks
    - [ ] Other: [Short answer]

## Course Planning
21. How many other courses are you taking this semester? [Short answer]

22. Weekly time available for this course (outside class):
    - [ ] 5-7 hours
    - [ ] 8-10 hours
    - [ ] 11-15 hours
    - [ ] More than 15 hours

23. Preferred project team size:
    - [ ] 2 people
    - [ ] 3 people
    - [ ] 4 people
    - [ ] 5 people

## Goals and Plans
24. Why are you taking this course? [Paragraph answer]

25. What specific skills do you hope to gain? [Paragraph answer]

26. Your plans after graduation:
    - [ ] Industry job
    - [ ] Graduate school
    - [ ] Starting a company
    - [ ] Not sure yet
    - [ ] Other: [Short answer]

## Additional Information
27. What concerns do you have about the course? [Paragraph answer]

28. Any other relevant background or information you'd like to share? [Paragraph answer]`;