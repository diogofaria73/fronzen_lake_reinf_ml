# Frozen Lake Q-Learning Project

Este projeto implementa um modelo de aprendizado por reforço para o ambiente Frozen Lake do Gymnasium, utilizando o algoritmo Q-Learning. A aplicação inclui visualização interativa com Pygame e geração de estatísticas de aprendizado.

## Requisitos

- Python 3.9 ou superior
- pipenv (para gerenciamento do ambiente virtual)

## Estrutura do Projeto

```
reinf_ml_puc/
│
├── src/
│   ├── training_code/        # Código de treinamento do modelo
│   │   └── q_learning.py     # Implementação do algoritmo Q-Learning
│   │
│   ├── visualization/        # Código de visualização
│   │   ├── pygame_visualizer.py  # Visualizador interativo usando Pygame
│   │   └── statistics_visualizer.py  # Gerador de gráficos de estatísticas
│   │
│   ├── results/              # Resultados do treinamento (modelos, gráficos)
│   │
│   └── main.py               # Script principal para execução
│
├── Pipfile                   # Configuração do pipenv
└── README.md                 # Este arquivo
```

## Instalação

1. Clone este repositório:
```bash
git clone <https://github.com/diogofaria73/fronzen_lake_reinf_ml.git>
cd reinf_ml_puc
```

2. Configure o ambiente virtual e instale as dependências:
```bash
pipenv install
```

## Uso

### Ativando o ambiente virtual

Antes de executar qualquer comando, ative o ambiente virtual:

```bash
pipenv shell
```

### Modos de Execução

O script principal `src/main.py` suporta vários modos de execução:

1. **Treinar um novo modelo**:
```bash
python src/main.py --mode train --episodes 10000
```

2. **Visualizar a performance do agente treinado**:
```bash
python src/main.py --mode visualize --model-path src/results/nome_do_modelo.pkl
```

3. **Gerar estatísticas de treinamento**:
```bash
python src/main.py --mode stats --model-path src/results/nome_do_modelo.pkl
```

4. **Fazer tudo (treinar, visualizar e gerar estatísticas)**:
```bash
python src/main.py --mode all
```

### Opções adicionais

- `--slippery`: Define se o gelo é escorregadio (mais difícil).
- `--map-size`: Tamanho do mapa (4 para 4x4, 8 para 8x8).
- `--learning-rate`: Taxa de aprendizado para o algoritmo Q-Learning.
- `--discount-factor`: Fator de desconto para recompensas futuras.
- `--fps`: Frames por segundo para a visualização.
- `--vis-episodes`: Número de episódios a serem visualizados.
- `--save-path`: Diretório para salvar modelos e resultados.

Exemplo de uso com mais parâmetros:
```bash
python src/main.py --mode all --episodes 5000 --map-size 8 --learning-rate 0.9 --discount-factor 0.97 --fps 10
```

## Sobre o Ambiente Frozen Lake

No ambiente "Frozen Lake", o agente precisa navegar de um ponto inicial até um objetivo (o buraco no gelo), evitando cair em buracos. Cada célula do mapa pode ser:

- **S**: Ponto inicial
- **F**: Gelo congelado (seguro)
- **H**: Buraco (perigo)
- **G**: Objetivo

O agente pode se mover em quatro direções: para cima, para baixo, para a esquerda e para a direita. Quando o parâmetro `is_slippery` está ativado, há uma chance do agente escorregar e mover-se em uma direção não intencional.

## Algoritmo Q-Learning

O algoritmo Q-Learning é usado para treinar o agente a tomar decisões ótimas. Ele aprende uma tabela de valores Q que associa cada par (estado, ação) com um valor de qualidade, indicando o quão boa é essa ação nesse estado.

A política final é derivada escolhendo a ação com o maior valor Q para cada estado. 