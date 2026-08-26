function selectMovie(element) {
    const selectedId = element.getAttribute('data-id');
    fetch("{% url 'design1' %}", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": "{{ csrf_token }}"
        },
        body: JSON.stringify({ selected_id: selectedId })
    }).then(() => window.location.reload());
}

