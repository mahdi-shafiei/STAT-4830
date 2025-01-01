function createSurveyForm() {
  // Create a new form
  const form = FormApp.create('STAT 4830: Student Background Survey');
  
  // Basic Information Section
  const basicSection = form.addSectionHeaderItem()
      .setTitle('Basic Information');
  
  form.addTextItem()
      .setTitle('1. Full Name')
      .setRequired(true);
  
  form.addTextItem()
      .setTitle('2. Penn Email')
      .setRequired(true);
  
  form.addTextItem()
      .setTitle('3. Major(s)/Minor(s)')
      .setRequired(true);
  
  form.addTextItem()
      .setTitle('4. Expected Graduation Year')
      .setRequired(true);
  
  form.addMultipleChoiceItem()
      .setTitle('5. Student Status')
      .setChoiceValues(['Undergraduate', 'Masters', 'PhD'])
      .showOtherOption(true)
      .setRequired(true);
  
  // Programming Background Section
  const progSection = form.addSectionHeaderItem()
      .setTitle('Programming Background');
  
  form.addMultipleChoiceItem()
      .setTitle('6. Rate your Python experience')
      .setChoiceValues([
        'No experience',
        'Basic (used in 1-2 courses)',
        'Intermediate (multiple courses/projects)',
        'Advanced (significant project work)',
        'Expert (professional experience)'
      ])
      .setRequired(true);
  
  form.addCheckboxItem()
      .setTitle('7. Which programming languages have you used?')
      .setChoiceValues(['Python', 'Java', 'C/C++', 'JavaScript', 'R', 'MATLAB'])
      .showOtherOption(true)
      .setRequired(true);
  
  // Programming comfort levels
  const progComfortGrid = form.addGridItem()
      .setTitle('8. Rate your comfort with:')
      .setRows([
        'Writing functions and classes',
        'Using Git/GitHub',
        'Debugging code',
        'Linux command line'
      ])
      .setColumns(['1', '2', '3', '4', '5'])
      .setRequired(true);
  
  form.addCheckboxItem()
      .setTitle('9. What development environments do you use?')
      .setChoiceValues([
        'VS Code',
        'Jupyter notebooks',
        'Google Colab',
        'Terminal + vim/emacs',
        'PyCharm'
      ])
      .showOtherOption(true);
  
  // Mathematics Background Section
  const mathSection = form.addSectionHeaderItem()
      .setTitle('Mathematics Background');
  
  form.addCheckboxItem()
      .setTitle('10. When did you last use these linear algebra concepts?')
      .setChoiceValues([
        'Matrix multiplication',
        'Vector spaces',
        'Eigenvalues/eigenvectors',
        'Matrix factorization',
        'Inner products',
        'None of these'
      ]);
  
  const calcGrid = form.addGridItem()
      .setTitle('11. Rate your comfort with calculus concepts:')
      .setRows([
        'Computing gradients',
        'Chain rule with vectors/matrices',
        'Optimization problems'
      ])
      .setColumns(['1', '2', '3', '4', '5'])
      .setRequired(true);
  
  form.addCheckboxItem()
      .setTitle('12. Are you familiar with these terms?')
      .setChoiceValues([
        'Jacobian',
        'Hessian',
        'Convexity',
        'Gradient descent',
        'None of these'
      ]);
  
  // Statistical Background Section
  const statSection = form.addSectionHeaderItem()
      .setTitle('Statistical Background');
  
  const statGrid = form.addGridItem()
      .setTitle('13. Rate your comfort with:')
      .setRows([
        'Probability distributions',
        'Maximum likelihood estimation',
        'Statistical modeling'
      ])
      .setColumns(['1', '2', '3', '4', '5'])
      .setRequired(true);
  
  // Machine Learning Experience Section
  const mlSection = form.addSectionHeaderItem()
      .setTitle('Machine Learning Experience');
  
  form.addCheckboxItem()
      .setTitle('14. Your ML experience includes:')
      .setChoiceValues([
        'Reading about ML concepts',
        'Completing online courses',
        'Course projects',
        'Research projects',
        'Industry experience',
        'No experience'
      ]);
  
  const mlGrid = form.addGridItem()
      .setTitle('15. Rate your experience with:')
      .setRows([
        'PyTorch',
        'Training neural networks',
        'Implementing ML algorithms'
      ])
      .setColumns(['1', '2', '3', '4', '5'])
      .setRequired(true);
  
  // Computing Resources Section
  const computingSection = form.addSectionHeaderItem()
      .setTitle('Computing Resources');
  
  form.addCheckboxItem()
      .setTitle('16. What GPU access do you have?')
      .setChoiceValues([
        'Personal computer with GPU',
        'Cloud GPU resources (AWS, GCP, etc.)',
        'University resources',
        'No access'
      ]);
  
  form.addCheckboxItem()
      .setTitle('17. Which AI tools do you use?')
      .setChoiceValues([
        'ChatGPT',
        'GitHub Copilot',
        'Claude',
        'None of these'
      ]);
  
  form.addMultipleChoiceItem()
      .setTitle('18. Have you installed Cursor IDE?')
      .setChoiceValues(['Yes', 'No', 'Not yet, but planning to']);
  
  // Research and Project Experience Section
  const researchSection = form.addSectionHeaderItem()
      .setTitle('Research and Project Experience');
  
  const researchGrid = form.addGridItem()
      .setTitle('19. Rate your experience with:')
      .setRows([
        'Reading academic papers',
        'Implementing algorithms from papers',
        'Giving technical presentations'
      ])
      .setColumns(['1', '2', '3', '4', '5'])
      .setRequired(true);
  
  form.addCheckboxItem()
      .setTitle('20. What project areas interest you?')
      .setChoiceValues([
        'Training/fine-tuning models',
        'Reproducibility studies',
        'Educational tools',
        'Optimization benchmarks'
      ])
      .showOtherOption(true);
  
  // Course Planning Section
  const planningSection = form.addSectionHeaderItem()
      .setTitle('Course Planning');
  
  form.addTextItem()
      .setTitle('21. How many other courses are you taking this semester?')
      .setRequired(true);
  
  form.addMultipleChoiceItem()
      .setTitle('22. Weekly time available for this course (outside class)')
      .setChoiceValues([
        '5-7 hours',
        '8-10 hours',
        '11-15 hours',
        'More than 15 hours'
      ])
      .setRequired(true);
  
  form.addMultipleChoiceItem()
      .setTitle('23. Preferred project team size')
      .setChoiceValues(['2 people', '3 people', '4 people', '5 people'])
      .setRequired(true);
  
  // Goals and Plans Section
  const goalsSection = form.addSectionHeaderItem()
      .setTitle('Goals and Plans');
  
  form.addParagraphTextItem()
      .setTitle('24. Why are you taking this course?')
      .setRequired(true);
  
  form.addParagraphTextItem()
      .setTitle('25. What specific skills do you hope to gain?')
      .setRequired(true);
  
  form.addMultipleChoiceItem()
      .setTitle('26. Your plans after graduation')
      .setChoiceValues([
        'Industry job',
        'Graduate school',
        'Starting a company',
        'Not sure yet'
      ])
      .showOtherOption(true)
      .setRequired(true);
  
  // Additional Information Section
  const additionalSection = form.addSectionHeaderItem()
      .setTitle('Additional Information');
  
  form.addParagraphTextItem()
      .setTitle('27. What concerns do you have about the course?');
  
  form.addParagraphTextItem()
      .setTitle('28. Any other relevant background or information you\'d like to share?');
  
  // Get the form URL
  Logger.log('Form URL: ' + form.getPublishedUrl());
  Logger.log('Form edit URL: ' + form.getEditUrl());
} 