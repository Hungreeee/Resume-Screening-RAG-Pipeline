import sys, os
sys.dont_write_bytecode = True

import pandas as pd
import streamlit as st
import numpy as np

def render(document_list: list, id_list: dict):
  retriever_message = st.expander(f"Click to see retriever's report 📄")
  with retriever_message:
    st.markdown("#### Retrieval results")
    button_columns = st.columns([0.2, 0.2, 0.2, 0.2, 0.2], gap="small")
    for index, document in enumerate(np.array(document_list)[:5]):
      with button_columns[index], st.popover(f"Resume {index + 1}"):
        st.markdown(f"""
                    ### Applicant ID {list(id_list.keys())[index]}

                    #### Ranker score: {round(list(id_list.values())[index], 4)}
                    """)
        st.markdown(document)
    st.markdown(f"""
    #### Other information
    - Top-k value: 5 resumes
    - Highest ranker score: {round(list(id_list.values())[0], 4)}
    - Lowest ranker score: {round(list(id_list.values())[-1], 4)}
    """)

if __name__ == "__main__":
  render(sys.argv[1], sys.argv[2])


