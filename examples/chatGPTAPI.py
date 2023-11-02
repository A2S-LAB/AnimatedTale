import openai

OPENAI_API_KEY = ""


openai.api_key = OPENAI_API_KEY

def createStory(front_prompt):
    back_prompt = """
                    Make a fairytale story of the following prompts and separate it by scenes.
                    number each scene (e.g. scene 1: story, scene 2: story) 
                    About 5 minutes long. 
                    The story should contain a good lesson for children to learn also stimulate childrens creativity.
                    Don't write anything other than scene number and the story you wrote. 
                    :
                  """
    model="gpt-3.5-turbo"
    messages=[
        {
            "role": "system",
            "content": "children's fairy tale author"
        },
        {
            "role": "user",
            "content": back_prompt + '\'' + front_prompt + '\''
        }
    ]
    answer = ''
    if front_prompt != '':
        response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            max_tokens=10
        )
        answer = response['choices'][0]['message']['content']
    
    return answer

if __name__ == '__main__':
    prompt = 'wizard of OZ, Wicked the Play. Wizards and animal friends.'

    story = createStory(prompt)
    print(story)