from wordcloud import WordCloud
import matplotlib.pyplot as plt
from itertools import cycle

# List of Gen AI evaluation frameworks
frameworks = [
    "LangSmith","RAGAS", "Deepeval", "LlamaIndex", "Lunary", "EvalGPT", "PromptLayer",
    "Trubrics", "EvalFlow", "ReEval", "GenAIBench", "OpenEval", "ParaEval",
    "EvalFarm", "AutoEval", "QAEval", "LLMEval", "T5Eval", "GenEval", "FastEval"
]

# Function to alternate between red and yellow
colors = cycle(["black","#BA0C2F", "#FFCC00"])

# Generate the word cloud with alternating colors
wordcloud = WordCloud(width=800, height=400, background_color="white", 
                      color_func=lambda *args, **kwargs: next(colors),
                      random_state=42).generate(" ".join(frameworks))

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

