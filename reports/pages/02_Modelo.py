import streamlit as st
import config
from src import get_data, train_model, generate_graphs
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error,
                             r2_score)
from sklearn.utils import estimator_html_repr
from PIL import Image

st.title('Modelo')

tab_modelagem_inicial, tab_resultados_iniciais, tab_conceitos, tab_variaveis, tab_simulacao, tab_deploy_producao = st.tabs(['Modelagem', "Resultados", 'Conceitos', "Importância das Variáveis", "Simulação", "Plano - Deploy em Produção"])

df_petroleo = get_data._get_modelling_data()

with tab_modelagem_inicial:
    st.markdown("""
        Iniciando a etapa de modelagem, optamos por experimentar o modelo Prophet, criado pela Meta/Facebook em 2017, 
        sendo um algoritmo de previsão de séries temporais que lida de forma eficiente com séries onde temos uma presença 
        forte e destacada de Sazonalidades, Feriados previamente conhecidoss e uma tendência de crescimento destacada.
        O mesmo define a série temporal tendo três componentes principais, tendência (g), sazonalidade (s) e feriados (h), 
        combinados na seguinte equação:

        `y(t)=g(t)+s(t)+h(t)+εt`, onde εt é o termo de erro, basicamente.
        
        Iniciaremos com um modelo simples, para verificar seu desempenho, e partiremos para conceitos mais elaborados, 
        chegando a um modelo hiperparametrizado, e com um desempenho superior para a aplicação.
                
    """)

    _model, X_test, pred, X_train, forecast_ = train_model._train_simple_prophet(df_petroleo)
    baseline_mape = round(mean_absolute_percentage_error(X_test['y'].values, pred['yhat'].values)*100, 3)
    baseline_r2 = round(r2_score(X_train['y'].values, forecast_['yhat'].values), 4)

with tab_resultados_iniciais:
    st.plotly_chart(
        plot_plotly(_model, forecast_.dropna()),
        use_container_width=True,
    )

    st.markdown(f"""
        De acordo com o gráfico acima, podemos ver que a previsão do modelo, embora com resultados interessantes,
        ainda carece de um ajuste melhor. 
                
        No período de teste, datado entre {min(pred.ds).date()} e {max(pred.ds).date()}, temos um erro médio absoluto percentual de 
        **{baseline_mape}%**,
        e um R2 (medida de ajuste na etapa de treinamento) de 
        **{baseline_r2}**
                
    """)

with tab_conceitos:
    st.markdown(f"""
        A partir dessa modelagem inicial, e da necessidade de analisarmos de forma mais direta o que está influenciando na previsão dos valores de preço de Petróleo, iremos utilizar alguns 
        conceitos mais avançados para melhorar o desempenho de nossa previsão:
                
        #### Variáveis externas
                
        Com a descoberta de variáveis de diferentes fontes que tem uma relação forte com nossa variável alvo, vamos aproveitá-las para obter um modelo potencialmente melhor, mas principalmente mais explicativo;
                
        ### XGBoost
        
        Um dos algoritmos de árvore conhecidos, se aproveita do resultado de diversas árvores de decisão, construídas de forma sequencial, onde cada nova árvore corrige o erro da árvore anterior, até que uma condição de parada seja alcançada.
        O algoritmo ainda aplica diversas penalidades de regularização, afim de evitar o overfitting, ou seja, uma adaptação muito forte aos dados de treinamento do modelo;

    """)
    st.image(config.DECISION_TREES,
                caption="Árvores de Decisão com Extreme Gradient Boosting (XGBoost)",
                width=700, 
        )

    st.markdown(f"""     
        #### Feature Importance

        Ao utilizarmos diversas variáveis para prever o valor do preço do Petróleo, podemos mensurar qual a importância de cada uma delas para a previsão do mesmo.

        No caso apresentado a seguir, apresentamos a importância através do "gain", ou seja, o erro no treinamento reduzido em cada divisão, em todas as árvores, estando mais relacionado a como as árvores operam individualmente.

        Comumente, podemos definir as variáveis com maior importância como os mais impactantes para a movimentação da variável alvo, tornando a análise útil para identificar onde podemos atuar em uma melhoria possível para a variável alvo, norteando ações que possam causar o maior impacto possível em menos tempo;       
       
    """)
    
    st.markdown(f"""            
        #### Pipelines
                
        São ferramentas úteis para normalizarmos o nosso processo de previsão, incluindo passos para evitarmos valores nulos, tratar variáveis com um pré-processamento, e até mesmo a própria execução do modelo ou seleção de melhores hiperparâmetros.
                
        #### "Serialização" do Pipeline
                
        Após elaborado o pipeline, salvamos o mesmo em um arquivo serializado, visando reutilizar o processo para previsão, e salvá-lo posteriormente em um catálogo de modelos.
                
        #### Deploy dos modelos
                
        Basicamente, colocar em produção o modelo, de forma a inferir o resultado a partir do mesmo, geralmente através de uma API, que recebe dados e aplica o pipeline salvo, conforme o último tópico.
                
        Para esses últimos 3 tópicos, comumente utilizamos ferramentas como o MLFlow, que possui formas de monitorar, servir os modelos em produção via API, criar projetos, empacotando códigos em um formato reprodutível em várias plataformas, e registrar/gerenciar modelos em uma espécie de repositório.
        Outra ferramenta comumente utilizada são Feature Stores, onde armazenamos variáveis em um format reutilizável em outros processos e modelos, compartilhando de forma padronizada e confiável esses dados. Tal ferramenta também está disponível via MLFlow.
    """)

with tab_variaveis:
    df_final = get_data._df_tree_modelling()
    dict_results = train_model._run_xgboost(df_final)
    df_importances = train_model._get_tree_importances(dict_results['pipeline'])
    st.markdown("""
    Para a análise de quais features mais importam, treinaremos um segundo modelo - chamado XGBoost, conforme explicado nos conceitos.
    Abaixo, vemos os passos do pipeline de previsão:
    """)
    st.image(config.PIPELINE,
                caption="Pipeline de Previsão usando XGBoost",
                width=680,
        )
    st.markdown(f"Para esse modelo, o mape ficou em **{dict_results['mape']}**!")

    st.plotly_chart(
        generate_graphs._plot_df_importances(df_importances),
        use_container_width=True
    )
    st.markdown("""
    Aqui, notamos a presença dominante das informações de exportação de combustíveis, de países diferentes, como Índia, Bolívia, Polônia, Sudão, Georgia, Alemanha, Israel, Dinamarca, Jordânia, Groenlândia, Canadá e Finlândia.
    <decorrer pq>.
                
    O valor médio da Dow jones também aponta como influente, sendo um dos indicadores apresentados anteriormente.
    """)
with tab_simulacao:
    st.markdown("Para efeito de simulação dos futuros preços do petróleo e do deploy de um modelo, vamos modificar duas das features organizadas pela importância do modelo de árvore (XGBoost), dado que possuam alguma importância no modelo:")
    col1, col2 = st.columns(2)
    with col1:
        option1 = st.selectbox(
            "Selecione a primeira:",
            (x.split('__')[1] for x in df_importances['Features'])
        )
    with col2:
        adjust1 = st.slider('Percentual (1)', -1.00, 1.00, 0.05)

    col1, col2 = st.columns(2)
    with col1:
        option2 = st.selectbox(
            "Selecione a Segunda:",
            (x.split('__')[1] for x in list(set(df_importances['Features'])-set([option1])))
        )
    with col2:
        adjust2 = st.slider('Percentual (2)', -1.00, 1.00, 0.05)

    st.divider()

    st.markdown('Para efeitos de simulação, vamos recuperar os valores dos últimos 3 anos de dados, e modificar conforme o solicitado, gerando as previsões a seguir:')
   
    # df_final = pd.DataFrame(imp.fit_transform(df_final), columns=df_final.columns).iloc[-10:]
    res = train_model.adjust_predict_data(
        df_final, 
        dict_cols = {
              option1:adjust1
            , option2:adjust2
        },
        _model=train_model._run_xgboost
    )
    st.plotly_chart(
        generate_graphs._plot_predictions(res['predictions']),
        use_container_width=True
    )
    st.markdown(f"A modificação das variáveis conforme selecionado, modificou em {res['mpe']} o valor do Petróleo, em média")


with tab_deploy_producao:
    st.markdown(f"""
    Antes de elaborar a arquitetura executamos a análise exploratória para compreender os dados e como automatizar o processo         
    """)
    image_eda = Image.open('reports/figures/eda.png')
    st.image(image_eda, caption='Fluxo análise exploratória de dados')

    image_arq = Image.open('reports/figures/arq.png')
    st.image(image_arq, caption='Arquitetura - modelo conceitual',)

    st.markdown("""
    Passos executados na ingestão, processamento, análise e predição
                
    - O ETL é acionado através do agendador Airflow
    - Os dados são capturados e persistidos (incremental) em banco de dados 
    - Pipeline com limpeza, normalização, geração e persistência das features é executado através do Airflow
    - O treinamento é realizado sempre que há mudanças nas variáveis/features ou desempenho. O modelo gerado é persistido num catálogo.
    - API gera predição com base no modelo salvo
    - Dashboard (e/ou outros clientes) consomem a predição gerada pela API
    """)

