/**
 * JavaScript for hyperparameter tuning functionality
 * Handles the enhanced hyperparameter tuning UI and integration with backend
 */

const HyperparameterTuningModule = (function() {
    // Private variables
    let _hyperparameterConfig = {};
    
    /**
     * Initialize the module
     */
    function init() {
        // Set up event handlers
        setupEventHandlers();
        
        // Initialize the hyperparameter configuration
        updateHyperparameterConfig();
    }
    
    /**
     * Set up all event handlers for hyperparameter tuning
     */
    function setupEventHandlers() {
        // Handle checkbox changes for hyperparameter tuning options
        const tuningCheckboxes = document.querySelectorAll('#hyperparameterTuningParams input[type="checkbox"]');
        tuningCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                updateHyperparameterConfig();
            });
        });
        
        // Listen for experiment type changes to initialize when hyperparameter tuning is selected
        const experimentTypeInputs = document.querySelectorAll('input[name="experiment_type"]');
        experimentTypeInputs.forEach(input => {
            input.addEventListener('change', function() {
                if (this.value === 'hyperparameter_tuning') {
                    updateHyperparameterConfig();
                }
            });
        });
    }
    
    /**
     * Update the hyperparameter configuration based on UI selections
     */
    function updateHyperparameterConfig() {
        _hyperparameterConfig = {
            tune_hidden_layers: document.getElementById('tuneHiddenLayers')?.checked || false,
            tune_learning_rate: document.getElementById('tuneLearningRate')?.checked || false,
            tune_activation: document.getElementById('tuneActivation')?.checked || false,
            tune_batch_size: document.getElementById('tuneBatchSize')?.checked || false,
            tune_dropout: document.getElementById('tuneDropout')?.checked || false,
            tune_optimizer: document.getElementById('tuneOptimizer')?.checked || false
        };
        
        console.log('Updated hyperparameter config:', _hyperparameterConfig);
    }
    
    /**
     * Get the current hyperparameter configuration
     * @returns {Object} The current hyperparameter configuration
     */
    function getHyperparameterConfig() {
        return _hyperparameterConfig;
    }
    
    /**
     * Prepare form data for submission
     * @param {FormData} formData - The form data to enhance
     * @returns {FormData} Enhanced form data with hyperparameter configuration
     */
    function prepareFormData(formData) {
        // Add hyperparameter configuration to form data
        if (formData.get('experiment_type') === 'hyperparameter_tuning') {
            Object.entries(_hyperparameterConfig).forEach(([key, value]) => {
                formData.set(key, value ? 'on' : 'off');
            });
        }
        
        return formData;
    }
    
    /**
     * Visualize hyperparameter tuning results
     * @param {Object} results - The hyperparameter tuning results
     */
    function visualizeResults(results) {
        if (!results || !results.ga_results || !results.pso_results) {
            console.error('Invalid hyperparameter tuning results');
            return;
        }
        
        // Create visualization containers if they don't exist
        const resultsContainer = document.getElementById('hyperparameterResults');
        if (!resultsContainer) {
            console.error('Results container not found');
            return;
        }
        
        // Clear previous results
        resultsContainer.innerHTML = '';
        
        // Create comparison chart
        const comparisonChart = document.createElement('div');
        comparisonChart.id = 'hyperparameterComparisonChart';
        comparisonChart.className = 'chart-container mb-4';
        resultsContainer.appendChild(comparisonChart);
        
        // Create performance chart
        const performanceChart = document.createElement('div');
        performanceChart.id = 'hyperparameterPerformanceChart';
        performanceChart.className = 'chart-container';
        resultsContainer.appendChild(performanceChart);
        
        // Request visualization from the server
        fetch('/api/visualize_hyperparameters', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                ga_results: results.ga_results,
                pso_results: results.pso_results
            })
        })
        .then(response => response.json())
        .then(data => {
            // Display the visualization
            if (data.comparison_chart) {
                document.getElementById('hyperparameterComparisonChart').innerHTML = 
                    `<img src="data:image/png;base64,${data.comparison_chart}" class="img-fluid" alt="Hyperparameter Comparison">`;
            }
            
            if (data.performance_chart) {
                document.getElementById('hyperparameterPerformanceChart').innerHTML = 
                    `<img src="data:image/png;base64,${data.performance_chart}" class="img-fluid" alt="Algorithm Performance">`;
            }
        })
        .catch(error => {
            console.error('Error visualizing hyperparameter results:', error);
        });
    }
    
    // Public API
    return {
        init: init,
        getHyperparameterConfig: getHyperparameterConfig,
        prepareFormData: prepareFormData,
        visualizeResults: visualizeResults
    };
})();

// Initialize the module when the DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    HyperparameterTuningModule.init();
});
