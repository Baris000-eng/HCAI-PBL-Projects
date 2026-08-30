document.addEventListener('DOMContentLoaded', () => {
    const list = document.getElementById('sortableList');
    let draggedItem = null;

    if (list) {
        // Handle drag start
        list.addEventListener('dragstart', (e) => {
            draggedItem = e.target.closest('.rank-item');
            if (draggedItem) {
                draggedItem.classList.add('dragging');
            }
        });

        // Handle drag end
        list.addEventListener('dragend', () => {
            if (draggedItem) {
                draggedItem.classList.remove('dragging');
                draggedItem = null;
            }
        });

        // Reorder items in real time as user drags
        list.addEventListener('dragover', (e) => {
            e.preventDefault();
            const afterElement = getDragAfterElement(list, e.clientY);
            if (draggedItem) {
                if (afterElement == null) {
                    list.appendChild(draggedItem);
                } else {
                    list.insertBefore(draggedItem, afterElement);
                }
            }
        });
    }
});

// Helper function to calculate the position based on vertical cursor coordinates
function getDragAfterElement(container, y) {
    const draggableElements = [...container.querySelectorAll('.rank-item:not(.dragging)')];
    return draggableElements.reduce((closest, child) => {
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;
        if (offset < 0 && offset > closest.offset) {
            return { offset: offset, element: child };
        } else {
            return closest;
        }
    }, { offset: Number.NEGATIVE_INFINITY }).element;
}

// Triggered by onclick="submitRanking()" in design2.html
function submitRanking() {
    const items = document.querySelectorAll('#sortableList .rank-item');
    const rankedIds = Array.from(items).map(item => parseInt(item.getAttribute('data-id')));

    fetch(DESIGN2_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": CSRF_TOKEN
        },
        body: JSON.stringify({
            ranked_ids: rankedIds
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            window.location.href = RECOMMENDATIONS_URL;
        }
    })
    .catch(error => console.error('Error submitting ranking:', error));
}