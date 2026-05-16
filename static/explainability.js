document.addEventListener("DOMContentLoaded", function () {
    try {
        const pdpLabelsRaw = document.getElementById("pdp-labels")?.textContent;
        const pdpDataRaw = document.getElementById("pdp-data")?.textContent;
        const aleLabelsRaw = document.getElementById("ale-labels")?.textContent;
        const aleDataRaw = document.getElementById("ale-data")?.textContent;

        if (!pdpLabelsRaw || !pdpDataRaw || !aleLabelsRaw || !aleDataRaw) {
            console.error("Critical DOM Error: Data target script elements could not be found.");
            return;
        }

        const gridLabels = JSON.parse(pdpLabelsRaw);
        const pdpData = JSON.parse(pdpDataRaw);
        const binCenters = JSON.parse(aleLabelsRaw);
        const aleData = JSON.parse(aleDataRaw);

        const colors = { 
            'Adelie': '#10b981', 
            'Chinstrap': '#3b82f6', 
            'Gentoo': '#f59e0b' 
        };
        const defaultColorList = ['#8b5cf6', '#ec4899', '#6366f1'];

        const pdpCanvas = document.getElementById('pdpChart');
        if (pdpCanvas) {
            new Chart(pdpCanvas, {
                type: 'line',
                data: {
                    labels: gridLabels,
                    datasets: Object.keys(pdpData).map((species, i) => ({
                        label: species,
                        data: pdpData[species],
                        borderColor: colors[species] || defaultColorList[i % defaultColorList.length],
                        backgroundColor: (colors[species] || defaultColorList[i % defaultColorList.length]) + '10',
                        tension: 0.15,
                        fill: false,
                        borderWidth: 2.5,
                        pointRadius: 2
                    }))
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top' },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Feature Grid Steps' } },
                        y: { title: { display: true, text: 'Marginal Probabilities' }, min: 0, max: 1 }
                    }
                }
            });
        }

        const aleCanvas = document.getElementById('aleChart');
        if (aleCanvas) {
            new Chart(aleCanvas, {
                type: 'line',
                data: {
                    labels: binCenters,
                    datasets: Object.keys(aleData).map((species, i) => ({
                        label: species,
                        data: aleData[species],
                        borderColor: colors[species] || defaultColorList[i % defaultColorList.length],
                        backgroundColor: (colors[species] || defaultColorList[i % defaultColorList.length]) + '10',
                        tension: 0.15,
                        fill: false,
                        borderWidth: 2.5,
                        pointRadius: 2
                    }))
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top' },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Local Quantile Bin Centers' } },
                        y: { title: { display: true, text: 'Centered Local Effect Δ' } }
                    }
                }
            });
        }

    } catch (error) {
        console.error("Chart Lifecycle Exception caught during processing:", error);
    }
});

function updateFormHiddenInputs() {
    const form = document.getElementById('controls-form');
    if (!form) return;
    
    const modelSel = document.getElementById('model_selector');
    const featSel = document.getElementById('feature_selector');
    
    if (modelSel) form.querySelector('input[name="model_type"]').value = modelSel.value;
    if (featSel) form.querySelector('input[name="feature"]').value = featSel.value;
}