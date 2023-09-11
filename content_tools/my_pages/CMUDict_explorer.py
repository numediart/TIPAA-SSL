import streamlit as st
import pandas as pd


from utils import formatted_cmudict_df as df
from utils import check_password

filters_df =  pd.DataFrame(columns=['Column', 'Operation', 'Value'])

# Define a function to apply the filters to the dataframe
def apply_filters(df, filters_df):
    for i,filter in filters_df.iterrows():
        mask = df[filter['Column']].str.__getattribute__(filter['Operation'])(filter['Value'])
        if filter['does/does not']!='does':
            mask=~mask
        df=df[mask]
    return df


import io
# Create a main function
def main(filters_df):
    # Create a list to store the filters
    
    st.title('CMU Dict explorer')

    if check_password():

        st.markdown("""
        This tool allows you to filter English words contained in [CMUDict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) based on criterea in text and in phonetics.
        You can find [here a spreadsheet](https://docs.google.com/spreadsheets/d/1aOEJyXFjbERq8vZ2qYMGsWnqpkIvzmL3Q4khBLMZ0Zc/edit?usp=sharing) that list the different CMU phonemes and their link to IPA and ARPABET and an example of word containing them.
        - First choose the number of filters (criterea) you want
        - Then build your filters
            - choose on what to apply a filter (text or formatted_phonetics)
            - if you want to include or exclude (because criterea is does/does not)
            - operation acn be: startswith, endswith, contains
            - what is the string (letters or sequence of phonemes)
        
        Example: if you wand to select words that finish in "-ed" and for which the phonetics ends in "D" like in "cleaned" or "moved", but not like in "guided" or "started", you will need three filters:
        - text | does | endswith | ed
        - formatted_phonetics | does | endswith | D
        - formatted_phonetics | does not | endswith | IH0_D
        """)
        
        # Get the number of rows to add from the user
        num_rows = st.number_input('Number of filters to add', min_value=1, max_value=10, value=1)

        
        # Add a form to allow the user to add new rows to the dataframe
        for i in range(num_rows):

            # Add a filter
            new_row = {}
            a, b, c, d = st.columns([1, 1, 1, 1])
            with a:
                new_row['Column'] = st.selectbox(f'Column (row {i+1})', ['text', 'formatted_phonetics'])
            with b:
                new_row['does/does not'] = st.selectbox(f'does/does not (row {i+1})', ['does', 'does not'])
            with c:
                new_row['Operation'] = st.selectbox(f'Operation (row {i+1})', ['startswith', 'endswith', 'contains'])
            with d:
                new_row['Value'] = st.text_input(f'Value (row {i+1})')
            filters_df = pd.concat([filters_df, pd.DataFrame([new_row])], ignore_index=True)


        
        st.header('Filters')
        st.dataframe(filters_df)

        
        # Apply the filters and display the result
        result = apply_filters(df, filters_df)

        st.header('Results')
        st.write(result)
        
        f = io.BytesIO()
        result.to_csv(f)
        d=st.download_button('Download list in CSV', f, file_name="list.csv")

# Run the main function
if __name__ == '__main__':
    main(filters_df)
