function toggleSeed(checkbox) {
    const input = document.getElementById('seed_input');
    if (checkbox.checked) {
        input.disabled = true;
        input.style.opacity = "0.5";
        input.value = ""; 
    } else {
        input.disabled = false;
        input.style.opacity = "1";
        input.value = "42";
    }
}

document.querySelector('form').addEventListener('submit', function(e) {
    const evaluation_metrics = document.querySelectorAll('input[name="eval_metrics"]');
    const is_checked = Array.from(evaluation_metrics).some(cb => cb.checked);

    if (!is_checked) {
        e.preventDefault(); 
        alert("Please select at least one evaluation metric before training the model!");
    }
});

function toggleIterationInput() {
    const modelSelect = document.querySelector('select[name="model_type"]');
    const iterInput = document.getElementById('iter_input');
    const iterContainer = document.getElementById('iter_container');
    
    if (!modelSelect || !iterInput || !iterContainer) return;

    const selectedModel = modelSelect.value;
    const needsIteration = ['svm', 'log_reg', 'sgdc'];
    
    if (needsIteration.includes(selectedModel)) {
        iterInput.disabled = false;
        iterContainer.style.opacity = "1";
        iterInput.style.cursor = "text";
    } else {
        iterInput.disabled = true;
        iterContainer.style.opacity = "0.5";
        iterInput.style.cursor = "not-allowed";
    }
}

window.onload = () => {
    toggleIterationInput();

    const modelSelect = document.querySelector('select[name="model_type"]');
    modelSelect.addEventListener('change', toggleIterationInput);
};