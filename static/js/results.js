/**
 * JavaScript for results visualization page
 * Handles loading experiment results, displaying charts, and parameter impact studies
 */

// Debug function to inspect objects
function debugObject(obj, name) {
    console.log('DEBUG ' + name + ':', obj);
    if (obj === null) return 'null';
    if (obj === undefined) return 'undefined';
    
    // Try to stringify the object
    try {
        return JSON.stringify(obj, null, 2);
    } catch (e) {
        return 'Error stringifying object: ' + e.message;
    }
}

$(document).ready(function() {
    console.log('Results page loaded, fetching results...');
    
    // Show loading indicator
    $('#resultsContent').hide();
    $('#noResultsMessage').hide();
    $('<div id="loadingResults" class="text-center mt-4"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">Loading results...</p></div>').insertAfter('#noResultsMessage');
    
    // Load experiment results
    $.ajax({
        url: '/api/get_results',
        type: 'GET',
        success: function(response) {
            console.log('API response received:', response);
            debugObject(response, 'Full API Response');
            
            // Remove loading indicator
            $('#loadingResults').remove();
            
            if (response.success) {
                // Show results content
                $('#resultsContent').show();
                $('#noResultsMessage').hide();
                
                // Display experiment info
                $('#experimentType').text(response.data.experiment_type || 'Unknown');
                // Use current date if not provided
                $('#experimentDate').text(new Date().toLocaleString());
                $('#datasetName').text('Current Dataset');
                
                // Display GA results if available
                if (response.data.has_ga_results && response.data.ga) {
                    $('#gaResultsContent').show();
                    $('#gaNoResults').hide();
                    
                    // Update GA metrics
                    $('#gaBestFitness').text(response.data.ga.best_fitness.toFixed(4));
                    $('#gaAccuracy').text(response.data.ga.test_metrics.accuracy.toFixed(4));
                    $('#gaPrecision').text(response.data.ga.test_metrics.precision.toFixed(4));
                    $('#gaRecall').text(response.data.ga.test_metrics.recall.toFixed(4));
                    $('#gaF1').text(response.data.ga.test_metrics.f1.toFixed(4));
                    $('#gaTrainingTime').text(response.data.ga.training_time.toFixed(2) + ' seconds');
                } else {
                    $('#gaResultsContent').hide();
                    $('#gaNoResults').show();
                }
                
                // Display PSO results if available
                if (response.data.has_pso_results && response.data.pso) {
                    $('#psoResultsContent').show();
                    $('#psoNoResults').hide();
                    
                    // Update PSO metrics
                    $('#psoBestFitness').text(response.data.pso.best_fitness.toFixed(4));
                    $('#psoAccuracy').text(response.data.pso.test_metrics.accuracy.toFixed(4));
                    $('#psoPrecision').text(response.data.pso.test_metrics.precision.toFixed(4));
                    $('#psoRecall').text(response.data.pso.test_metrics.recall.toFixed(4));
                    $('#psoF1').text(response.data.pso.test_metrics.f1.toFixed(4));
                    $('#psoTrainingTime').text(response.data.pso.training_time.toFixed(2) + ' seconds');
                } else {
                    $('#psoResultsContent').hide();
                    $('#psoNoResults').show();
                }
                
                // Display comparison charts if both algorithms were run
                if (response.data.has_comparison && response.data.comparison) {
                    $('#algorithmComparisonSection').show();
                    $('#comparisonContent').show();
                    $('#comparisonNoResults').hide();
                    
                    // Display performance comparison chart
                    if (response.data.comparison.comparison_plot) {
                        $('#comparisonChart').attr('src', 'data:image/png;base64,' + response.data.comparison.comparison_plot);
                    }
                    
                    // Display hyperparameter comparison chart
                    if (response.data.comparison.param_importance_plot) {
                        $('#hyperparameterComparisonChart').attr('src', 'data:image/png;base64,' + response.data.comparison.param_importance_plot);
                    }
                    
                    // Display convergence plot
                    if (response.data.comparison.convergence_plot) {
                        $('#convergenceChart').attr('src', 'data:image/png;base64,' + response.data.comparison.convergence_plot);
                        $('#convergenceContent').show();
                        $('#convergenceNoResults').hide();
                    } else {
                        $('#convergenceContent').hide();
                        $('#convergenceNoResults').show();
                    }
                    
                    // Populate comparison metrics table
                    if (response.data.ga && response.data.pso) {
                        // Accuracy comparison
                        const gaAcc = response.data.ga.test_metrics.accuracy;
                        const psoAcc = response.data.pso.test_metrics.accuracy;
                        $('#compAccuracyGA').text(gaAcc.toFixed(4));
                        $('#compAccuracyPSO').text(psoAcc.toFixed(4));
                        $('#compAccuracyDiff').text(Math.abs(gaAcc - psoAcc).toFixed(4));
                        
                        // Precision comparison
                        const gaPrecision = response.data.ga.test_metrics.precision;
                        const psoPrecision = response.data.pso.test_metrics.precision;
                        $('#compPrecisionGA').text(gaPrecision.toFixed(4));
                        $('#compPrecisionPSO').text(psoPrecision.toFixed(4));
                        $('#compPrecisionDiff').text(Math.abs(gaPrecision - psoPrecision).toFixed(4));
                        
                        // Recall comparison
                        const gaRecall = response.data.ga.test_metrics.recall;
                        const psoRecall = response.data.pso.test_metrics.recall;
                        $('#compRecallGA').text(gaRecall.toFixed(4));
                        $('#compRecallPSO').text(psoRecall.toFixed(4));
                        $('#compRecallDiff').text(Math.abs(gaRecall - psoRecall).toFixed(4));
                        
                        // F1 comparison
                        const gaF1 = response.data.ga.test_metrics.f1;
                        const psoF1 = response.data.pso.test_metrics.f1;
                        $('#compF1GA').text(gaF1.toFixed(4));
                        $('#compF1PSO').text(psoF1.toFixed(4));
                        $('#compF1Diff').text(Math.abs(gaF1 - psoF1).toFixed(4));
                        
                        // Training time comparison
                        const gaTime = response.data.ga.training_time;
                        const psoTime = response.data.pso.training_time;
                        $('#compTimeGA').text(gaTime.toFixed(2) + 's');
                        $('#compTimePSO').text(psoTime.toFixed(2) + 's');
                        $('#compTimeDiff').text(Math.abs(gaTime - psoTime).toFixed(2) + 's');
                    }
                    
                    // Populate convergence details
                    if (response.data.ga && response.data.ga.history) {
                        const gaHistory = response.data.ga.history;
                        if (gaHistory.best_fitness && gaHistory.best_fitness.length > 0) {
                            const initialFitness = gaHistory.best_fitness[0];
                            const finalFitness = gaHistory.best_fitness[gaHistory.best_fitness.length - 1];
                            $('#gaInitialFitness').text(initialFitness.toFixed(4));
                            $('#gaFinalFitness').text(finalFitness.toFixed(4));
                            $('#gaImprovement').text(((finalFitness - initialFitness) * 100 / initialFitness).toFixed(2) + '%');
                            
                            // Calculate convergence speed (generations to reach 90% of final fitness)
                            let genTo90Percent = gaHistory.best_fitness.length;
                            const targetFitness = initialFitness + 0.9 * (finalFitness - initialFitness);
                            for (let i = 0; i < gaHistory.best_fitness.length; i++) {
                                if (gaHistory.best_fitness[i] >= targetFitness) {
                                    genTo90Percent = i + 1;
                                    break;
                                }
                            }
                            $('#gaConvergenceSpeed').text(genTo90Percent + ' generations');
                            
                            // Early stopping info
                            const earlyStop = gaHistory.best_fitness.length < response.data.ga_params.num_generations;
                            $('#gaEarlyStopping').text(earlyStop ? 'Yes, at generation ' + gaHistory.best_fitness.length : 'No');
                        }
                    }
                    
                    if (response.data.pso && response.data.pso.history) {
                        const psoHistory = response.data.pso.history;
                        if (psoHistory.best_fitness && psoHistory.best_fitness.length > 0) {
                            const initialFitness = psoHistory.best_fitness[0];
                            const finalFitness = psoHistory.best_fitness[psoHistory.best_fitness.length - 1];
                            $('#psoInitialFitness').text(initialFitness.toFixed(4));
                            $('#psoFinalFitness').text(finalFitness.toFixed(4));
                            $('#psoImprovement').text(((finalFitness - initialFitness) * 100 / initialFitness).toFixed(2) + '%');
                            
                            // Calculate convergence speed (iterations to reach 90% of final fitness)
                            let iterTo90Percent = psoHistory.best_fitness.length;
                            const targetFitness = initialFitness + 0.9 * (finalFitness - initialFitness);
                            for (let i = 0; i < psoHistory.best_fitness.length; i++) {
                                if (psoHistory.best_fitness[i] >= targetFitness) {
                                    iterTo90Percent = i + 1;
                                    break;
                                }
                            }
                            $('#psoConvergenceSpeed').text(iterTo90Percent + ' iterations');
                            
                            // Early stopping info
                            const earlyStop = psoHistory.best_fitness.length < response.data.pso_params.num_iterations;
                            $('#psoEarlyStopping').text(earlyStop ? 'Yes, at iteration ' + psoHistory.best_fitness.length : 'No');
                        }
                    }
                } else {
                    $('#algorithmComparisonSection').hide();
                    $('#comparisonContent').hide();
                    $('#comparisonNoResults').show();
                    $('#convergenceContent').hide();
                    $('#convergenceNoResults').show();
                }
            } else {
                // Show no results message
                $('#resultsContent').hide();
                $('#noResultsMessage').show();
                console.log('No results available or error in response:', response);
            }
        },
        error: function(xhr, status, error) {
            // Remove loading indicator
            $('#loadingResults').remove();
            
            // Log error details
            console.error('API Error:', status, error);
            console.error('Response:', xhr.responseText);
            
            // Show error message
            $('#resultsContent').hide();
            $('#noResultsMessage').show().html(
                '<div class="alert alert-danger">'+
                '<i class="fas fa-exclamation-circle me-2"></i>'+
                'Error loading results: ' + error + '<br>'+
                'Please check the console for more details or try again later.'+
                '</div>'
            );
        }
    });
    
    // Add parameter value field
    $('#addParameterValue').on('click', function() {
        const newField = `
            <div class="input-group mb-2">
                <input type="number" class="form-control parameter-value" step="0.01" required>
                <button type="button" class="btn btn-outline-danger remove-parameter-value">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        $('#parameterValuesContainer').append(newField);
    });
    
    // Remove parameter value field
    $('#parameterValuesContainer').on('click', '.remove-parameter-value', function() {
        $(this).closest('.input-group').remove();
    });
    
    // Update parameter options based on selected algorithm
    $('#parameterAlgorithm').on('change', function() {
        const algorithm = $(this).val();
        const parameterSelect = $('#parameterName');
        
        // Clear existing options
        parameterSelect.empty();
        
        // Add algorithm-specific options
        if (algorithm === 'ga') {
            parameterSelect.append('<option value="population_size">Population Size</option>');
            parameterSelect.append('<option value="num_generations">Number of Generations</option>');
            parameterSelect.append('<option value="mutation_rate">Mutation Rate</option>');
            parameterSelect.append('<option value="crossover_rate">Crossover Rate</option>');
            parameterSelect.append('<option value="tournament_size">Tournament Size</option>');
            parameterSelect.append('<option value="elite_size">Elite Size</option>');
        } else if (algorithm === 'pso') {
            parameterSelect.append('<option value="num_particles">Number of Particles</option>');
            parameterSelect.append('<option value="num_iterations">Number of Iterations</option>');
            parameterSelect.append('<option value="inertia_weight">Inertia Weight</option>');
            parameterSelect.append('<option value="cognitive_coefficient">Cognitive Coefficient</option>');
            parameterSelect.append('<option value="social_coefficient">Social Coefficient</option>');
            parameterSelect.append('<option value="max_velocity">Maximum Velocity</option>');
        }
    });
    
    // Run parameter study
    $('#parameterStudyForm').on('submit', function(e) {
        e.preventDefault();
        
        // Get form data
        const algorithm = $('#parameterAlgorithm').val();
        const parameterName = $('#parameterName').val();
        const runsPerValue = $('#runsPerValue').val();
        
        // Get parameter values
        const parameterValues = [];
        $('.parameter-value').each(function() {
            parameterValues.push(parseFloat($(this).val()));
        });
        
        // Create form data for AJAX request
        const requestData = new FormData();
        requestData.append('algorithm', algorithm);
        requestData.append('parameter_name', parameterName);
        requestData.append('runs_per_value', runsPerValue);
        requestData.append('parameter_values', JSON.stringify(parameterValues));
        
        // Show loading state
        const submitButton = $(this).find('button[type="submit"]');
        submitButton.prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Running...');
        
        // Send AJAX request
        $.ajax({
            url: '/api/parameter_impact_study',
            type: 'POST',
            data: requestData,
            processData: false,
            contentType: false,
            success: function(response) {
                // Reset button
                submitButton.prop('disabled', false).html('<i class="fas fa-play me-2"></i>Run Parameter Study');
                
                if (response.success) {
                    // Show results
                    $('#parameterStudyResults').show();
                    
                    // Display parameter impact chart
                    $('#parameterImpactChart').attr('src', 'data:image/png;base64,' + response.plot);
                    
                    // Display parameter results table
                    const table = $('#parameterResultsTable tbody');
                    table.empty();
                    
                    for (let i = 0; i < response.parameter_values.length; i++) {
                        table.append(`
                            <tr>
                                <td>${response.parameter_values[i]}</td>
                                <td>${response.avg_fitness_scores[i].toFixed(4)}</td>
                                <td>${response.best_fitness_scores[i].toFixed(4)}</td>
                            </tr>
                        `);
                    }
                } else {
                    alert('Error: ' + response.error);
                }
            },
            error: function(xhr) {
                // Reset button
                submitButton.prop('disabled', false).html('<i class="fas fa-play me-2"></i>Run Parameter Study');
                
                const response = xhr.responseJSON || {};
                alert('Error: ' + (response.error || 'Failed to run parameter study'));
            }
        });
    });
});
