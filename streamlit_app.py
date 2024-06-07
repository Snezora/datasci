import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
from joblib import load
plt.ion()

import warnings
warnings.filterwarnings("ignore")

st.title("Online Payments Fraud Detection")

scaler = load('scaler.joblib')
y_test = load('y_test.joblib')
X_test_scaled = load('X_test_scaled.joblib')

if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'encoder' not in st.session_state:
    st.session_state['encoder'] = None
    

def main():
    global model
    model = None
    
    # Load the model into session state
    if st.button('Load Logistic Regression Model'):
        st.session_state['model'] = load('lr_model.joblib')
        st.session_state['encoder'] = load('lr_encoder.joblib')
        model = "lr"
        print("Logistic Regression Model loaded successfully!")
        
    if st.button('Load Decision Tree Model'):
        st.session_state['model'] = load('dt_model.joblib')
        st.session_state['encoder'] = load('dt_encoder.joblib')
        model = "dt"
        print("Decision Tree Model loaded successfully!")

    if st.session_state['model'] is not None:
        st.write('Model Loaded')
        
        # Create separate input sections for each feature
        step = st.number_input('Enter Step:', value=0)
        type = st.selectbox('Select Type:', ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
        amount = st.number_input('Enter Amount:', value=0)
        oldbalanceOrg = st.number_input('Enter Old Balance Orig:', value=0)
        newbalanceOrig = st.number_input('Enter New Balance Orig:', value=0)
        oldbalanceDest = st.number_input('Enter Old Balance Dest:', value=0)
        newbalanceDest = st.number_input('Enter New Balance Dest:', value=0)
        isFlaggedFraud = st.toggle("Is the transaction flagged as fraud?", value=0)
        
        # Create a button to make predictions
        if st.button('Make Prediction'):
            # Create a DataFrame with the input values
            new_data_df = pd.DataFrame({'step': [step], 'type': [type], 'amount': [amount], 'oldbalanceOrg': [oldbalanceOrg], 'newbalanceOrig': [newbalanceOrig], 'oldbalanceDest': [oldbalanceDest], 'newbalanceDest': [newbalanceDest], 'isFlaggedFraud' : [isFlaggedFraud]})

            # Encode categorical values
            new_data_df['type'] = st.session_state['encoder'].transform(new_data_df['type'])

            # Scale the data
            new_data_scaled = scaler.transform(new_data_df)

            # Make prediction
            prediction = st.session_state['model'].predict(new_data_scaled)
            print("Prediction:", prediction)
            st.write('Prediction:', prediction)
            
            if (prediction == 0 and model == "lr"):
                st.markdown('''After calculation by the model, it has determined that the transaction is :green[NOT Fraud]''')
                st.markdown(''':blue-background[Although do be warned, this model is not that accurate. If desire an accurate model, please switch over to the Decision Tree Model]''')
            elif (prediction == 1 and model == "lr"):
                st.markdown('''After calculation by the model, it has determined that the transaction is :red[Fraud]''')
                st.markdown(''':blue-background[Although do be warned, this model is not that accurate. If desire an accurate model, please switch over to the Decision Tree Model]''')
            elif (prediction == 0 and model == "dt"):
                st.markdown('''After calculation by the model, it has determined that the transaction is :green[NOT Fraud]''')
            elif (prediction == 1 and model == "dt"):
                st.markdown('''After calculation by the model, it has determined that the transaction is :red[Fraud]''')
            else:
                st.markdown('''Please select a model''')


            
            # Display model metrics
            st.write('Model Metrics:')
            st.write('Accuracy:', accuracy_score(y_test, st.session_state['model'].predict(X_test_scaled)))
            st.write('Precision:', precision_score(y_test,st.session_state['model'].predict(X_test_scaled)))
            st.write('Recall:', recall_score(y_test, st.session_state['model'].predict(X_test_scaled)))
            st.write('F1 Score:', f1_score(y_test, st.session_state['model'].predict(X_test_scaled)))
            st.write('ROC-AUC Score:', roc_auc_score(y_test, st.session_state['model'].predict_proba(X_test_scaled)[:, 1]))

            # Display confusion matrix
            st.write('Confusion Matrix:')
            st.write(confusion_matrix(y_test, st.session_state['model'].predict(X_test_scaled)))

            # Display ROC-AUC curve
            st.write('ROC-AUC Curve:')
            y_pred_proba = st.session_state['model'].predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC-AUC Curve')
            plt.legend(loc='best')
            st.pyplot(plt)
            
            
if __name__ == '__main__':
    main()