import tkinter as tk
from tkinter import scrolledtext
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer from Hugging Face
model_name = "gpt2"  # You can use "gpt2-medium", "gpt2-large" for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
gpt2_model.eval()  # Set the model to evaluation mode

# Text generation function for GPT-2
def generate_gpt2_text(prompt, max_length=150, temperature=1.0, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output = gpt2_model.generate(input_ids, 
                                      max_length=max_length, 
                                      num_return_sequences=1, 
                                      no_repeat_ngram_size=2,
                                      temperature=temperature,
                                      top_k=top_k,
                                      top_p=top_p,
                                      do_sample=True)
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# GUI Application Class
class TextGenerationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPT-2 Text Generator")
        self.root.geometry("600x400")

        # Label
        self.label = tk.Label(root, text="Enter Seed Text:", font=("Helvetica", 14))
        self.label.pack(pady=10)

        # Entry for Seed Text
        self.seed_text_entry = tk.Entry(root, font=("Helvetica", 12), width=60)
        self.seed_text_entry.pack(pady=10)

        # Button to generate text
        self.generate_button = tk.Button(root, text="Generate Text", font=("Helvetica", 12), command=self.generate_text)
        self.generate_button.pack(pady=10)

        # Scrolled Text box for showing the generated text
        self.result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=10, font=("Helvetica", 12))
        self.result_text.pack(pady=10)

    def generate_text(self):
        seed_text = self.seed_text_entry.get()  # Get seed text from entry widget
        if not seed_text:
            self.result_text.insert(tk.END, "Please enter a seed text!\n")
            return

        # Generate text using GPT-2
        generated_text = generate_gpt2_text(seed_text, max_length=200)
        
        # Display the generated text in the scrolled text widget
        self.result_text.delete(1.0, tk.END)  # Clear previous result
        self.result_text.insert(tk.END, generated_text + "\n")  # Show the new result

# Create the main window
root = tk.Tk()

# Create the text generation application
app = TextGenerationApp(root)

# Start the GUI event loop
root.mainloop()
