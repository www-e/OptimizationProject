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
                
                // Display comparison chart if both algorithms were run
                if (response.data.has_comparison && response.data.comparison) {
                    $('#algorithmComparisonSection').show();
                    if (response.data.comparison.comparison_plot) {
                        $('#comparisonChart').attr('src', 'data:image/png;base64,' + response.data.comparison.comparison_plot);
                    }
                } else {
                    $('#algorithmComparisonSection').hide();
                }
                
                // Display convergence charts if available
                if (response.data.has_comparison && response.data.comparison) {
                    $('#convergenceSection').show();
                    
                    if (response.data.has_ga_results && response.data.comparison.convergence_plot) {
                        $('#gaConvergenceSection').show();
                        $('#gaConvergenceChart').attr('src', 'data:image/png;base64,' + response.data.comparison.convergence_plot);
                    } else {
                        $('#gaConvergenceSection').hide();
                    }
                    
                    if (response.data.has_pso_results && response.data.comparison.convergence_plot) {
                        $('#psoConvergenceSection').show();
                        $('#psoConvergenceChart').attr('src', 'data:image/png;base64,' + response.data.comparison.convergence_plot);
                    } else {
                        $('#psoConvergenceSection').hide();
                    }
                    
                    if (response.ga_results && response.pso_results) {
                        $('#combinedConvergenceSection').show();
                        $('#combinedConvergenceChart').attr('src', 'data:image/png;base64,' + response.convergence_charts.combined);
                    } else {
                        $('#combinedConvergenceSection').hide();
                    }
                } else {
                    $('#convergenceSection').hide();
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
