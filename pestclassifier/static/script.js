document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('file-input');
    const resultDiv = document.getElementById('result');
    const modal = document.getElementById('myModal');
    const modalImage = document.getElementById('modal-image');
    const modalResult = document.getElementById('modal-result');
    const closeModal = document.getElementsByClassName('close')[0];

    if (fileInput.files.length === 0) {
        resultDiv.textContent = 'Please select a file.';
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        if (data.error) {
            modalResult.textContent = `Error: ${data.error}`;
        } else {
            modalResult.textContent = `Classification Result: ${data.class_name}`;
        }

        // Display the uploaded image in the modal
        const reader = new FileReader();
        reader.onload = function(e) {
            modalImage.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Show the modal
        modal.style.display = 'block';

    } catch (error) {
        modalResult.textContent = `Error: ${error.message}`;
    }

    // Close the modal when the user clicks on <span> (x)
    closeModal.onclick = function() {
        modal.style.display = 'none';
    };

    // Close the modal when the user clicks anywhere outside of the modal
    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    };
});
