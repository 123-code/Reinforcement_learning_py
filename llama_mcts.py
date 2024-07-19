from groq import Groq
import numpy as np
import random
import math
import re
from sympy import symbols,integrate,sin,cos,diff

def create_response(user_prompt: str):
    client = Groq(api_key="")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert mathematician, you break down mathematical problems into simpler pieces, making it easier for them to be solved.
                """
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        model="llama3-70b-8192",
        temperature = 1
    )
    response = chat_completion.choices[0].message.content
    print(response)
    return response

def rate_answer(question: str, answer: str):
    prompt = f"""
    Pregunta: {question}
    Respuesta: {answer}
    
    Eres un experto en diversos temas. Basado en esta pregunta y respuesta, dame una crítica de la respuesta a la pregunta. Solo una crítica, no la respuesta. Luego provee un rating del 1-100 de acuerdo a que tan bien crees que la pregunta fue respondida.
    La respuesta va en este formato: Critica:<crítica>, Rating:<rating>
    """
    critica_y_rating = create_response(prompt)
    try:
        match = re.search(r'Rating:\s*(\d+)', critica_y_rating)
        if match:
            rating = int(match.group(1))
            if rating > 95:
                rating = 95
            rating = float(rating) / 100
        else:
            raise ValueError("Rating not found")
    except Exception as e:
        print(f"error: {e}")
        rating = 0.0
    print(rating)
    return rating

def get_critique(question: str, answer: str):
    prompt = f"""
    Pregunta: {question}
    Respuesta: {answer}
    
    Por favor dime cómo mejorarías esta respuesta a la pregunta indicada. Da instrucciones breves y concisas.
    """
    return create_response(prompt)

def improve_answer(question: str, answer: str):
    critique = get_critique(question, answer)
    prompt = f"""
    Pregunta: {question}
    Respuesta: {answer}
    Crítica: {critique}
    
    Por favor provee una respuesta mejorada a la pregunta indicada, basado en la crítica.
    Para cada respuesta, dame cuál fue tu proceso de razonamiento. Usa el formato:
    Razonamiento: <Proceso de razonamiento>
    Verificación: <Verificación de la respuesta>
    Respuesta final: <la respuesta final verificada>
    """
    return create_response(prompt)

def calculate_answer(eq:str):
    x = symbols('x')
    f = eq
    g = integrate(f, x)    
    h = diff(g,x)
    if h == f:
        return True
    else:
        return False

max_children = 5

class Node:
    def __init__(self, question, answer, parent=None):
        self.question = question
        self.answer = answer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) >= max_children

    def get_best_child(self, exploration_rate=1.41):
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float('inf')
            else:
                weight = (child.value / child.visits) + exploration_rate * math.sqrt((2 * math.log(self.visits) / child.visits))
            choices_weights.append(weight)
        return self.children[np.argmax(choices_weights)]

    def add_child(self, child_node):
        self.children.append(child_node)

class MCTS:
    def __init__(self, question, seed_answers, iterations=5):
        self.question = question
        self.seed_answers = seed_answers
        self.iterations = iterations
        self.root = Node(question, seed_answers)

    def select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.get_best_child()
        return node

    def expand(self, node):
        for j in range(max_children - len(node.children)):
            child_node = Node(self.question, node.answer, parent=node)
            node.add_child(child_node)
            critique = get_critique(self.question, child_node.answer)
            improved = improve_answer(self.question, child_node.answer)
            child_node.answer = improved
        return random.choice(node.children)

    def search(self):
        for x in range(self.iterations):
            node = self.select(self.root)

            if not node.is_fully_expanded():
                node = self.expand(node)

            reward = self.simulate(node)
            self.backpropagate(node, reward)
        return self.root.get_best_child().answer

    def simulate(self, node):
        rating = rate_answer(self.question, node.answer)
        return rating

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

question = "Integra la función: f(x) = 2sin(13x) + 4cos(13x)"
seed_answer = create_response(question)
mcts = MCTS(question, seed_answer, iterations=5)
best_answer = mcts.search()
print(f"best answer: {best_answer}")


