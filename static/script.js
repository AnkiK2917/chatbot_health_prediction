// Function to send the message
function sendMessage() {
    const userInput = document.getElementById("userInput").value;

    if (userInput) {
        const chatbox = document.getElementById("chatbox");
        const userMessage = `<div class="chat-message"><span class="user">${userInput}</span></div>`;
        chatbox.innerHTML += userMessage;

        // Fetch response from the server
        fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'msg=' + encodeURIComponent(userInput)
        })
        .then(response => response.json())  // Parse the response as JSON
        .then(data => {
            // Access the 'response' field from the response data
            const botMessage = `<div class="chat-message"><span class="bot">${data.response}</span></div>`;
            chatbox.innerHTML += botMessage;
            chatbox.scrollTop = chatbox.scrollHeight;  // Scroll to the bottom of the chatbox
        })
        .catch(error => {
            console.error('Error:', error);
        });

        // Clear the input field after sending the message
        document.getElementById("userInput").value = '';
    }
}

// Event listener for Enter key
document.getElementById("userInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        event.preventDefault(); // Prevents newline in input field
        sendMessage(); // Call sendMessage function
    }
});
