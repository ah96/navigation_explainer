#!/usr/bin/env python

import rospy
import speech_recognition as sr
import pyttsx3
import openai
from std_msgs.msg import String

# Initialize the recognizer
recognizer = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set your OpenAI API key
openai.api_key = '---'

def listen_and_respond():
    # Initialize the ROS node
    rospy.init_node('tiago_chatbot_node', anonymous=True)
    
    # Publisher to announce that the robot is speaking
    pub = rospy.Publisher('/tiago_chatbot/speaking', String, queue_size=10)
    
    # Main loop
    while not rospy.is_shutdown():
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
        
        try:
            # Convert speech to text
            question = recognizer.recognize_google(audio)
            print(f"Question: {question}")
            
            # Generate response from ChatGPT-4
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                max_tokens=150
            ).choices[0].message['content'].strip()
            
            print(f"Response: {response}")
            
            # Convert text response to speech
            engine.say(response)
            engine.runAndWait()
            
            # Publish the response to ROS
            pub.publish(response)
        
        except sr.UnknownValueError:
            print("Could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    try:
        listen_and_respond()
    except rospy.ROSInterruptException:
        pass


