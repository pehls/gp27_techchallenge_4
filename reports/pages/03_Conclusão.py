import streamlit as st

st.title('Conclusão')

st.markdown("""
    Com base nos dados obtidos, podemos verificar alguns pontos interessantes:
    - O uso de energia e o consumo de combustíveis fósseis de muitos países tem uma correlação forte ou muito forte com o preço do petróleo no decorrer dos anos, sendo a maioria delas positiva, o que indica que carregam o mesmo sentido, quando o preço está em alta, temos um alto consumo e uso, quando em baixa, os mesmos diminuem;
    - Alguns países, como Angola, tem uma forte dependência do petróleo, tendo em vista uma alta porcentagem de sua economia derivada da exportação, e um uso e consumo extremamente elevados;
    - Dentre as principais crises do petróleo, muitas delas são derivadas de conflitos armados, o que reflete em correlações elevadamente positivas na média dos valores comparando ao petróleo;
    - Na análise de *feature importance* destacamos 2: A primeira delas (e mais influente) a exportação/consumo de combustíveis fósseis, com um foco e importância maior na Índia (3º maior consumidor de petróleo no mundo);
    - A 2ª variável destacada e que reforça a interconexão entre os mercados financeiros e de commodities, é evidenciada pelo impacto influente do valor médio da Dow Jones na variação do preço do petróleo.
""")
