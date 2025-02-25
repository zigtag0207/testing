import os
import sys


try:
   from langchain_openai import OpenAI
   from langchain.chains import ConversationChain
   from langchain.memory import ConversationBufferMemory
except ImportError:
   print("Installing required packages...")
   os.system(f"{sys.executable} -m pip install langchain-openai langchain-community")
   from langchain_openai import OpenAI
   from langchain.chains import ConversationChain
   from langchain.memory import ConversationBufferMemory

# checking
def initialize_chat():
   """Initializes the conversation with OpenAI and memory buffer."""
   try:
       import os
       openai_api_key = os.getenv("OPENAI_API_KEY")
       if not openai_api_key:
           raise ValueError("Missing OPENAI_API_KEY. Set it as an environment variable.")
      
       llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
       memory = ConversationBufferMemory()
       conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
       return conversation
   except Exception as e:
       print(f"Error initializing chat: {e}")
       sys.exit(1)


def chat_loop():
   """Runs a chat loop for user interaction."""
   conversation = initialize_chat()
  
   print("ChatGPT-like Assistant (Type 'quit' to exit)")
   print("-" * 50)
  
   while True:
       user_input = input("\nYou: ").strip()
       if user_input.lower() == 'quit':
           print("\nGoodbye!")
           break
      
       try:
           response = conversation.predict(input=user_input)
           print("\nAssistant:", response)
       except Exception as e:
           print(f"\nError occurred: {str(e)}")


if __name__ == "__main__":
   chat_loop()
