import re
import requests

from huggingface_hub import InferenceClient
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


TOMORROW_CODE_MAP = {
    0: "Unknown", 1000: "Clear", 1001: "Cloudy", 1100: "Mostly Clear",
    1101: "Partly Cloudy", 1102: "Mostly Cloudy", 2000: "Fog",
    2100: "Light Fog", 3000: "Light Wind", 3001: "Wind", 3002: "Strong Wind",
    4000: "Drizzle", 4001: "Rain", 4200: "Light Rain", 4201: "Heavy Rain",
    5000: "Snow", 5001: "Flurries", 5100: "Light Snow", 5101: "Heavy Snow",
    6000: "Freezing Drizzle", 6001: "Freezing Rain", 6200: "Light Freezing Rain",
    6201: "Heavy Freezing Rain", 7000: "Ice Pellets", 7101: "Heavy Ice Pellets",
    7102: "Light Ice Pellets", 8000: "Thunderstorm"
}
class CalculatorTool:
    def add(self, a, b):
        return a + b
    def multiply(self, a, b):
        return a * b

class WeatherTool:
    def __init__(self, api_key):
        self.api_key = api_key

    # Realtime endpoint for simplicity
    def get_weather(self, city):
        url = f"https://api.tomorrow.io/v4/weather/realtime?location={city}&units=metric&apikey={self.api_key}"
        try:
            res = requests.get(url, timeout=20)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            return f"Error fetching weather: {e}"

        try:
            vals = data.get("data", {}).get("values", {})
            temp = vals.get("temperature", "N/A")
            wcode = vals.get("weatherCode", 0)
            wx = TOMORROW_CODE_MAP.get(int(wcode) if str(wcode).isdigit() else 0, "Unknown")
            return f"The weather in {city} is {wx} with {temp}°C."
        except Exception as e:
            return f"Error reading weather data: {e}"

class SearchTool:
    def __init__(self):
        self.runner = DuckDuckGoSearchRun()
    def search(self, query):
        try:
            return self.runner.run(query)
        except Exception as e:
            return f"Search error: {e}"
class CalculatorAgent:
    def __init__(self):
        self.tool = CalculatorTool()
    #Has its own CalculatorTool.

    def perform_task(self, request):
        if any(word in request.lower() for word in ["add", "plus", "+", "multiply", "times", "*"]):
            numbers = list(map(int, re.findall(r'-?\d+', request)))
            if len(numbers) < 2:
                return "Not enough numbers for calculation."
            if any(word in request.lower() for word in ["add", "plus", "+"]):
                return self.tool.add(numbers[0], numbers[1])
            elif any(word in request.lower() for word in ["multiply", "times", "*"]):
                return self.tool.multiply(numbers[0], numbers[1])
        return None

class WeatherAgent:
    def __init__(self, api_key):
        self.tool = WeatherTool(api_key)
      #Has its own WeatherTool.

    def perform_task(self, request):
        if "weather" in request.lower():
            city_match = re.search(r'weather in ([\w\s]+)', request, re.IGNORECASE)
            if city_match:
                city = city_match.group(1).strip()
                return self.tool.get_weather(city)
            else:
                return "Please specify a city for the weather."
        return None

class SearchAgent:
    def __init__(self):
        self.tool = SearchTool()
    def perform_task(self, request):
        if "search" in request.lower() or "find" in request.lower():
            q = request.replace("search", "").replace("find", "").strip()
            return self.tool.search(q if q else request)
        return None

# =========================
# ✅ LangChain: HF Runnable (Zephyr via chat_completion ✅)
# =========================
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=os.getenv("HF_TOKEN"))

class HFClientRunnable(Runnable):
    def invoke(self, input, config=None, **kwargs):
        # Prefer chat mode (Zephyr supports conversational)
        text = input.to_string() if hasattr(input, "to_string") else str(input)
        try:
            resp = client.chat_completion(
                messages=[{"role": "user", "content": text}],
                max_tokens=256,
                temperature=0.7,
            )
            choice = resp.choices[0]
            msg = getattr(choice, "message", choice.get("message", {}))
            content = getattr(msg, "content", msg.get("content", ""))
            if content:
                return content
        except Exception as e1:
            # Fallback to text_generation if provider allows
            try:
                return client.text_generation(prompt=text, max_new_tokens=256)
            except Exception as e2:
                return f"LLM error (chat/text): {e1} | {e2}"

hf_runnable = HFClientRunnable()

# Tool: DuckDuckGo Search for the chain
search_tool = DuckDuckGoSearchRun()

# Prompt with optional memory
prompt = PromptTemplate.from_template(
    """You are a helpful assistant. Answer the user's question. If the question needs current information, use this search result too:
Search Result: {search_result}

Conversation so far:
{chat_history}

User: {question}
Answer:"""
)

# Memory setup
memory = ConversationBufferMemory(return_messages=True)

# ChatAgent using LangChain chain (Prompt + Runnable + Parser + Memory + Search)
class ChatAgent:
    def __init__(self, prompt, runnable, memory, search_tool):
        self.prompt = prompt
        self.runnable = runnable
        self.memory = memory
        self.search_tool = search_tool

    def perform_task(self, question):
        # Check if search is needed
        if any(x in question.lower() for x in ["current", "today", "weather", "latest", "news"]):
            try:
                search_result = self.search_tool.run(question)
            except Exception:
                search_result = "N/A"
        else:
            search_result = "N/A"

        chat_history = self.memory.load_memory_variables({}).get("history", "")

        input_dict = {
            "search_result": search_result,
            "chat_history": chat_history,
            "question": question
        }

        chain = self.prompt | self.runnable | StrOutputParser()
        response = chain.invoke(input_dict)

        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(response)

        return response

# =========================
# ✅ Master Agent (Coordinator)
# =========================
class MasterAgent:
    def __init__(self, weather_api_key):
        self.agents = [
            CalculatorAgent(),
            WeatherAgent(weather_api_key),
            SearchAgent()
        ]
        self.chat_agent = ChatAgent(prompt, hf_runnable, memory, search_tool)

    def perform_task(self, request):
        # Try specialized agents
        for agent in self.agents:
            result = agent.perform_task(request)
            if result is not None:
                return result
        # Fallback to Chat (LangChain LLM chain)
        return self.chat_agent.perform_task(request)