// Round tracking and total rounds limit
const TOTAL_ROUNDS = 5;
let currentRound = parseInt(sessionStorage.getItem('design1_round') || '1');

// Update the round indicator when the DOM content is loaded
document.addEventListener("DOMContentLoaded", () => {
    const indicator = document.getElementById('round-indicator');
    if (indicator) {
        indicator.innerText = `Round ${currentRound} of ${TOTAL_ROUNDS}`;
    }
});

// Triggered by onclick="selectMovie(this)" in design1.html
function selectMovie(selectedElement) {
    const chosenId = parseInt(selectedElement.getAttribute('data-id'));
    const allCards = Array.from(document.querySelectorAll('.movie-card'));
    const otherCard = allCards.find(card => parseInt(card.getAttribute('data-id')) !== chosenId);
    const otherId = parseInt(otherCard.getAttribute('data-id'));

    fetch(DESIGN1_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": CSRF_TOKEN
        },
        body: JSON.stringify({
            chosen_id: chosenId,
            other_id: otherId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            if (currentRound >= TOTAL_ROUNDS) {
                // If 5 rounds are completed, clear session storage and redirect to recommendations page
                sessionStorage.removeItem('design1_round');
                window.location.href = RECOMMENDATIONS_URL;
            } else {
                // Increment round count, update session storage, and reload the page for the next pair
                sessionStorage.setItem('design1_round', currentRound + 1);
                location.reload();
            }
        }
    })
    .catch(error => console.error('Error submitting choice:', error));
}