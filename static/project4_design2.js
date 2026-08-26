const list = document.getElementById('sortableList');
let dragItem = null;

list.addEventListener('dragstart', (e) => {
    dragItem = e.target;
    e.target.classList.add('dragging');
});

list.addEventListener('dragend', (e) => {
    e.target.classList.remove('dragging');
});

list.addEventListener('dragover', (e) => {
    e.preventDefault();
    const afterElement = getDragAfterElement(list, e.clientY);
    if (afterElement == null) {
        list.appendChild(dragItem);
    } else {
        list.insertBefore(dragItem, afterElement);
    }
});

function getDragAfterElement(container, y) {
    const elements = [...container.querySelectorAll('.rank-item:not(.dragging)')];
    return elements.reduce((closest, child) => {
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;
        if (offset < 0 && offset > closest.offset) {
            return { offset: offset, element: child };
        } else {
            return closest;
        }
    }, { offset: Number.NEGATIVE_INFINITY }).element;
}

function submitRanking() {
    const ranking = [...list.querySelectorAll('.rank-item')].map(item => item.dataset.id);
    fetch("{% url 'design2' %}", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": "{{ csrf_token }}"
        },
        body: JSON.stringify({ ranking: ranking })
    }).then(() => alert("Rankings submitted successfully!"));
}