// State configuration parameters
let currentSampleIndex = null;
let currentSampleSimulatedLabel = null;
let categoryLabels = [];

document.addEventListener("DOMContentLoaded", () => {
    loadBaseMetrics();
    initLearningToDefer();
    loadNextActiveLearningSample();

    // Event listeners
    document.getElementById("threshold-slider").addEventListener("input", (e) => {
        document.getElementById("threshold-val").innerText = e.target.value;
        evaluateDeferralSystem(e.target.value);
    });

    document.getElementById("btn-use-expert").addEventListener("click", () => {
        submitQuery(currentSampleSimulatedLabel);
    });
});

// Load base metrics
async function loadBaseMetrics() {
    try {
        const res = await fetch("/project3/api/metrics/");
        const data = await res.json();
        console.log(data);
        
        document.getElementById("text-classifier-acc").innerText = (data.baseline_accuracy * 100).toFixed(2) + "%";
        document.getElementById("text-expert-acc").innerText = (data.expert_accuracy * 100).toFixed(2) + "%";
    } catch (err) {
        console.error("Error obtaining system baseline metrics:", err);
    }
}

// Learning to Defer Loop caller
async function evaluateDeferralSystem(threshold) {
    try {
        const res = await fetch("/project3/api/learning_to_defer/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ threshold: parseFloat(threshold) })
        });
        const data = await res.json();
        
        document.getElementById("defer-system-acc").innerText = (data.system_accuracy * 100).toFixed(2) + "%";
        document.getElementById("defer-rate").innerText = (data.deferral_rate * 100).toFixed(1) + "%";
        document.getElementById("deferred-count").innerText = `${data.deferred_count} / ${data.total_count}`;
    } catch (err) {
        console.error("Error parsing deferral strategy response:", err);
    }
}

function initLearningToDefer() {
    const currentVal = document.getElementById("threshold-slider").value;
    evaluateDeferralSystem(currentVal);
}

// Load Active Learning Sample
async function loadNextActiveLearningSample() {
    try {
        const res = await fetch("/project3/api/active_learning/next/");
        const data = await res.json();
        
        currentSampleIndex = data.sample_index;
        currentSampleSimulatedLabel = data.simulated_expert_label;
        categoryLabels = data.categories;

        document.getElementById("sample-text-box").innerText = data.text;
        document.getElementById("expert-recom").innerText = categoryLabels[data.simulated_expert_label];

        // Build Interactive buttons for Human Override Expert choices
        const humanOptionsContainer = document.getElementById("human-options");
        humanOptionsContainer.innerHTML = "";
        
        categoryLabels.forEach((catName, index) => {
            const btn = document.createElement("button");
            btn.className = "btn btn-secondary";
            btn.innerText = `Class ${index}: ${catName}`;
            btn.onclick = () => submitQuery(index);
            humanOptionsContainer.appendChild(btn);
        });

    } catch (err) {
        console.error("Error gathering active learning batch pipeline updates:", err);
    }
}

// Submit choice to loop cycle updates
async function submitQuery(labelSelected) {
    if (currentSampleIndex === null) {
        return;
    }
    
    try {
        const res = await fetch("/project3/api/active_learning/query/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sample_index: currentSampleIndex,
                label: parseInt(labelSelected)
            })
        });

        if(!res.ok) {
            throw new Error("Labeling task is completed. No more samples are available in the active learning pool.");
        }

        const data = await res.json();

        const manualLabelCount = data.total_labeled_count - 2000;
        
        // Update tracker UI metrics outputs
        document.getElementById("al-model-acc").innerText = (data.current_accuracy * 100).toFixed(2) + "%";
        document.getElementById("al-manual-query-num").innerText = manualLabelCount;
        document.getElementById("al-query-count").innerText = data.total_labeled_count;
        
        // Advance queue to next active calculation query item
        loadNextActiveLearningSample();
    } catch (err) {
        console.error("Error setting Active Learning label submission:", err);
    }
}

