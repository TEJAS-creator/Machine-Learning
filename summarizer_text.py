from transformers import pipeline
summarizer = pipeline("summarization")
article = """
Essay on One Piece Anime

One Piece is one of the greatest and most celebrated anime series of all time. Created by Eiichiro Oda, the story has captured the hearts of millions across the globe. The anime adaptation by Toei Animation began in 1999 and continues to run successfully even today, making it one of the longest-running series in history. What makes One Piece special is not just its length but also the depth of its storytelling, world-building, and emotional power.
At the center of the story is Monkey D. Luffy, a cheerful and determined young pirate who dreams of becoming the Pirate King. Luffy gains his powers after eating a Devil Fruit called the Gomu Gomu no Mi, which makes his body stretch like rubber. Though this ability is strange, it becomes his greatest weapon in battles against powerful enemies. However, the fruit also takes away his ability to swim, a huge weakness for a pirate. Despite this, Luffy’s spirit and determination make him unstoppable.
The adventure begins when Luffy sets out to find the legendary treasure known as the “One Piece.” This treasure, hidden at the end of the dangerous Grand Line, is said to grant unimaginable wealth and the title of Pirate King to whoever claims it. Luffy’s goal is not motivated by greed but by the desire for freedom, adventure, and proving himself to the world. His dream inspires not only himself but also the people around him.
"""

summary = summarizer(article,max_length=150,min_length=50, do_sample=False)
summary[0]["summary_text"]
