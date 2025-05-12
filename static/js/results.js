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
    
    // Event handlers for expand/collapse all buttons
    $('#expandAllCards').on('click', function() {
        $('.patient-card-summary .expand-details-btn').each(function() {
            const collapseElement = $($(this).data('bs-target'));
            if (!collapseElement.hasClass('show')) {
                $(this).click();
            }
        });
    });
    
    $('#collapseAllCards').on('click', function() {
        $('.collapse.show').each(function() {
            const collapseId = $(this).attr('id');
            $(`[data-bs-target="#${collapseId}"]`).click();
        });
    });
    
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
                    
                    // Add medical interpretation if available
                    if (response.data.ga.test_metrics.medical_interpretation) {
                        // Add a new row for medical interpretation if it doesn't exist
                        if ($('#gaMedicalInterpretation').length === 0) {
                            $('#gaResultsTable').append('<tr><th>Medical Interpretation</th><td id="gaMedicalInterpretation"></td></tr>');
                        }
                        $('#gaMedicalInterpretation').text(response.data.ga.test_metrics.medical_interpretation);
                    }
                    
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
                    
                    // Add medical interpretation if available
                    if (response.data.pso.test_metrics.medical_interpretation) {
                        // Add a new row for medical interpretation if it doesn't exist
                        if ($('#psoMedicalInterpretation').length === 0) {
                            $('#psoResultsTable').append('<tr><th>Medical Interpretation</th><td id="psoMedicalInterpretation"></td></tr>');
                        }
                        $('#psoMedicalInterpretation').text(response.data.pso.test_metrics.medical_interpretation);
                    }
                    
                    $('#psoTrainingTime').text(response.data.pso.training_time.toFixed(2) + ' seconds');
                } else {
                    $('#psoResultsContent').hide();
                    $('#psoNoResults').show();
                }
                
                // Display patient assessments if available
                if (response.data.patient_assessments && response.data.patient_assessments.length > 0) {
                    $('#patientAssessmentsContent').show();
                    $('#patientAssessmentsNoResults').hide();
                    
                    // Clear existing patient cards
                    $('#patientCards').empty();
                    
                    // Calculate risk distribution for analytics
                    let totalPatients = response.data.patient_assessments.length;
                    let highRiskCount = 0;
                    let moderateRiskCount = 0;
                    let lowRiskCount = 0;
                    
                    // Count patients by risk level
                    response.data.patient_assessments.forEach(function(patient) {
                        if (patient.risk_level === 'High') {
                            highRiskCount++;
                        } else if (patient.risk_level === 'Moderate to High' || patient.risk_level === 'Moderate') {
                            moderateRiskCount++;
                        } else {
                            lowRiskCount++;
                        }
                    });
                    
                    // Update analytics summary
                    $('#totalPatients').text(totalPatients);
                    $('#highRiskCount').text(highRiskCount);
                    $('#moderateRiskCount').text(moderateRiskCount);
                    $('#lowRiskCount').text(lowRiskCount);
                    
                    // Calculate percentages for progress bars
                    const highRiskPercent = Math.round((highRiskCount / totalPatients) * 100);
                    const moderateRiskPercent = Math.round((moderateRiskCount / totalPatients) * 100);
                    const lowRiskPercent = Math.round((lowRiskCount / totalPatients) * 100);
                    
                    // Update progress bars
                    $('#highRiskBar').css('width', highRiskPercent + '%').attr('aria-valuenow', highRiskPercent).text(highRiskPercent + '%');
                    $('#moderateRiskBar').css('width', moderateRiskPercent + '%').attr('aria-valuenow', moderateRiskPercent).text(moderateRiskPercent + '%');
                    $('#lowRiskBar').css('width', lowRiskPercent + '%').attr('aria-valuenow', lowRiskPercent).text(lowRiskPercent + '%');
                    
                    // Create patient cards
                    response.data.patient_assessments.forEach(function(patient) {
                        // Determine card color based on risk level
                        let cardColorClass = 'border-success';
                        let riskBadgeClass = 'bg-success';
                        
                        if (patient.risk_level === 'High') {
                            cardColorClass = 'border-danger';
                            riskBadgeClass = 'bg-danger';
                        } else if (patient.risk_level === 'Moderate to High' || patient.risk_level === 'Moderate') {
                            cardColorClass = 'border-warning';
                            riskBadgeClass = 'bg-warning';
                        }
                        
                        // Create risk factors list
                        let riskFactorsList = '';
                        if (patient.risk_factors && patient.risk_factors.length > 0) {
                            riskFactorsList = '<ul class="list-group list-group-flush mb-3">';
                            patient.risk_factors.forEach(function(factor) {
                                riskFactorsList += `
                                    <li class="list-group-item p-2">
                                        <div class="d-flex justify-content-between">
                                            <span><strong>${factor.factor}</strong> (${factor.value.toFixed(2)})</span>
                                            <span class="badge ${factor.severity === 'High' ? 'bg-danger' : 'bg-warning'} text-white">${factor.severity}</span>
                                        </div>
                                        <small class="text-muted">${factor.medical_implication}</small>
                                    </li>
                                `;
                            });
                            riskFactorsList += '</ul>';
                        } else {
                            riskFactorsList = '<p class="text-muted">No significant risk factors identified.</p>';
                        }
                        
                        // Create risk patterns list
                        let riskPatternsList = '';
                        if (patient.risk_patterns && patient.risk_patterns.length > 0) {
                            riskPatternsList = '<div class="mt-3">';
                            riskPatternsList += '<h6 class="mb-2">Identified Risk Patterns:</h6>';
                            patient.risk_patterns.forEach(function(pattern) {
                                riskPatternsList += `
                                    <div class="alert alert-warning p-2 mb-2">
                                        <strong>${pattern.pattern}</strong> (${pattern.severity})
                                        <p class="small mb-1">${pattern.description}</p>
                                        <p class="small mb-0 text-danger">${pattern.medical_implication}</p>
                                    </div>
                                `;
                            });
                            riskPatternsList += '</div>';
                        }
                        
                        // Create biomarkers table
                        let biomarkersTable = '<table class="table table-sm mt-3">';
                        biomarkersTable += '<thead><tr><th>Biomarker</th><th>Value</th></tr></thead><tbody>';
                        
                        for (const [key, value] of Object.entries(patient.biomarkers)) {
                            // Format the biomarker name for display
                            const displayName = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            biomarkersTable += `<tr><td>${displayName}</td><td>${value.toFixed(2)}</td></tr>`;
                        }
                        
                        biomarkersTable += '</tbody></table>';
                        
                        // Create the patient card - now with collapsible content
                        const patientCard = `
                            <div class="col-md-6 col-lg-3 mb-3">
                                <div class="card shadow-sm ${cardColorClass}" id="patientCard${patient.patient_id}">
                                    <!-- Compact header - always visible -->
                                    <div class="card-header d-flex justify-content-between align-items-center py-2">
                                        <h6 class="mb-0 fw-bold">Patient #${patient.patient_id}</h6>
                                        <span class="badge ${riskBadgeClass} ms-2">${patient.risk_level}</span>
                                    </div>
                                    
                                    <!-- Compact risk display - always visible -->
                                    <div class="card-body py-2 patient-card-summary">
                                        <div class="d-flex justify-content-between align-items-center">
                                            <span class="small">Risk Probability:</span>
                                            <strong class="ms-2">${(patient.predicted_risk_probability * 100).toFixed(1)}%</strong>
                                        </div>
                                        <div class="progress mt-1" style="height: 8px;">
                                            <div class="progress-bar ${riskBadgeClass}" role="progressbar" 
                                                style="width: ${(patient.predicted_risk_probability * 100).toFixed(1)}%" 
                                                aria-valuenow="${(patient.predicted_risk_probability * 100).toFixed(1)}" 
                                                aria-valuemin="0" aria-valuemax="100">
                                            </div>
                                        </div>
                                        <div class="text-center mt-2">
                                            <button class="btn btn-sm btn-outline-primary expand-details-btn w-100" 
                                                data-bs-toggle="collapse" data-bs-target="#patientDetails${patient.patient_id}" 
                                                aria-expanded="false" aria-controls="patientDetails${patient.patient_id}">
                                                <i class="fas fa-chevron-down me-1"></i>View Details
                                            </button>
                                        </div>
                                    </div>
                                    
                                    <!-- Collapsible detailed content -->
                                    <div class="collapse" id="patientDetails${patient.patient_id}">
                                        <div class="card-body pt-0 border-top">
                                            <!-- Risk Factors Section -->
                                            <div class="mt-3">
                                                <h6 class="fw-bold text-secondary mb-2">Risk Factors:</h6>
                                                ${riskFactorsList}
                                            </div>
                                            
                                            <!-- Risk Patterns Section -->
                                            ${riskPatternsList}
                                            
                                            <!-- Biomarkers Section -->
                                            <div class="accordion mt-3" id="biomarkerAccordion${patient.patient_id}">
                                                <div class="accordion-item border-0 shadow-sm">
                                                    <h2 class="accordion-header" id="biomarkerHeading${patient.patient_id}">
                                                        <button class="accordion-button collapsed bg-light" type="button" data-bs-toggle="collapse" 
                                                            data-bs-target="#biomarkerCollapse${patient.patient_id}" aria-expanded="false" 
                                                            aria-controls="biomarkerCollapse${patient.patient_id}">
                                                            <i class="fas fa-vial me-2 text-primary"></i>Biomarker Details
                                                        </button>
                                                    </h2>
                                                    <div id="biomarkerCollapse${patient.patient_id}" class="accordion-collapse collapse" 
                                                        aria-labelledby="biomarkerHeading${patient.patient_id}" data-bs-parent="#biomarkerAccordion${patient.patient_id}">
                                                        <div class="accordion-body p-0">
                                                            ${biomarkersTable}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            
                                            <!-- Recommendation Section -->
                                            <div class="mt-3 p-2 bg-light rounded">
                                                <h6 class="fw-bold text-secondary mb-1">Medical Recommendation:</h6>
                                                <p class="mb-0 small">${patient.recommendation}</p>
                                            </div>
                                            
                                            <!-- Collapse button -->
                                            <div class="text-center mt-3">
                                                <button class="btn btn-sm btn-outline-secondary w-100" 
                                                    data-bs-toggle="collapse" data-bs-target="#patientDetails${patient.patient_id}" 
                                                    aria-expanded="true" aria-controls="patientDetails${patient.patient_id}">
                                                    <i class="fas fa-chevron-up me-1"></i>Collapse Details
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                        
                        // Add the card to the container
                        $('#patientCards').append(patientCard);
                    });
                } else {
                    $('#patientAssessmentsContent').hide();
                    $('#patientAssessmentsNoResults').show();
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
