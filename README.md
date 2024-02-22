1. "Investigate the use of BERT":   
BERT is a powerful model, its  resource requirements and complexity may not be suitable for  web applications, especially if real-time or reactive processing  is required.
 Low latency feedback is needed for BERT.
Other models such as LSTM or SVM can provide a more effective balance between accuracy and speed for web applications.

2.  Continuous integration and Continuous delivery/deployment (CICD):   
CICD is not necessary for our web application due to factors such as : limited resources, time constraints, infrequent changes, , budget constraints, and when developed by a single developer or a small team.

3.  Related work papers not enough:    
Since we are doing a Web app , 4 related works were enough to explain the choice of using LSTM and/or SVM for our application.

4. TF-IDF isn't needed (no context):   
TF-IDF has its own goodness, especially when dealing with text data.
While models like BERT can capture more nuanced relationships, TF-IDF features can still provide valuable insights, especially in situations where Contextual integration may not be  available or necessary. 
TFIDF can leverage both context modeling and traditional feature engineering for optimal results.

5. Building Dataset or Datasets Ready?:   
We are not going to Build a dataset form scratch since our choice to use a pre-existing Twitter dataset aligns strategically with the primary scope of our project, which is aspect-based hate speech detection. Twitter, being a microblogging platform, has been widely recognized as a rich source of diverse and real-time textual data that encompasses various aspects of language, including offensive, sexist, and racist content. By leveraging an established Twitter dataset, we ensure that our model is trained on a vast and relevant corpus that reflects the nuances and dynamics of online hate speech within the targeted scope.

6. Smaller Scope:   
The web application can be designed with flexibility to accommodate regional variations. By incorporating user feedback and continuously updating the model based on the evolving language landscape, the application can adapt to different contexts

7. How to detect newly invented hate speech? (suggestion: use automl):   
While AutoML is a valuable tool, it may not be the only solution.
Regular model updates, continuous monitoring of online trends, and user reporting mechanisms can be used to identify and incorporate new hate speech this will fix the problem new hate words rather than using automl.
 A collaborative, adaptive approach involving human moderation and automated tools may be more effective in managing emerging language patterns instead of using automl.
