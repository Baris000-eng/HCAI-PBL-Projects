let myChart;
const rawData = JSON.parse(document.getElementById('data-json').textContent);

function drawChart() {
    const xSelect = document.getElementById('xColumn');
    const ySelect = document.getElementById('yColumn');
    
    const xCol = xSelect.value;
    const yCol = ySelect.value;
    const xLabel = xSelect.options[xSelect.selectedIndex].text;
    const yLabel = ySelect.options[ySelect.selectedIndex].text;

    const formattedData = rawData.map(d => ({ x: d[xCol], y: d[yCol] }));

    if (myChart) myChart.destroy();

    const ctx = document.getElementById('myChart').getContext('2d');
    myChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: `${xLabel} vs ${yLabel}`,
                data: formattedData,
                backgroundColor: '#275CB2'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

document.addEventListener('DOMContentLoaded', drawChart);