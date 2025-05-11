/**
 * JavaScript for algorithm configuration page
 * Handles form submissions, configuration saving/loading, and tab interactions
 */

$(document).ready(function() {
    // Save GA configuration
    $('#saveGaConfig').on('click', function() {
        const configName = prompt('Enter a name for this configuration:');
        if (!configName) return;
        
        // Collect form data
        const formData = new FormData($('#gaForm')[0]);
        
        // Convert value range to array
        const valueRangeMin = parseFloat(formData.get('value_range_min'));
        const valueRangeMax = parseFloat(formData.get('value_range_max'));
        
        // Create config object
        const configData = {
            population_size: parseInt(formData.get('population_size')),
            num_generations: parseInt(formData.get('num_generations')),
            chromosome_type: formData.get('chromosome_type'),
            value_range: [valueRangeMin, valueRangeMax],
            selection_method: formData.get('selection_method'),
            tournament_size: parseInt(formData.get('tournament_size')),
            crossover_rate: parseFloat(formData.get('crossover_rate')),
            mutation_rate: parseFloat(formData.get('mutation_rate')),
            elitism: formData.get('elitism') === 'on',
            elite_size: parseInt(formData.get('elite_size')),
            crossover_type: formData.get('crossover_type'),
            adaptive_mutation: formData.get('adaptive_mutation') === 'on'
        };
        
        // Create form data for AJAX request
        const requestData = new FormData();
        requestData.append('config_type', 'ga');
        requestData.append('config_name', configName);
        requestData.append('config_data', JSON.stringify(configData));
        
        // Send AJAX request
        $.ajax({
            url: '/api/save_config',
            type: 'POST',
            data: requestData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.success) {
                    alert('Configuration saved successfully!');
                    // Refresh page to update saved configurations list
                    location.reload();
                } else {
                    alert('Error: ' + response.error);
                }
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                alert('Error: ' + (response.error || 'Failed to save configuration'));
            }
        });
    });
    
    // Save PSO configuration
    $('#savePsoConfig').on('click', function() {
        const configName = prompt('Enter a name for this configuration:');
        if (!configName) return;
        
        // Collect form data
        const formData = new FormData($('#psoForm')[0]);
        
        // Convert value range to array
        const valueRangeMin = parseFloat(formData.get('value_range_min'));
        const valueRangeMax = parseFloat(formData.get('value_range_max'));
        
        // Create config object
        const configData = {
            num_particles: parseInt(formData.get('num_particles')),
            num_iterations: parseInt(formData.get('num_iterations')),
            value_range: [valueRangeMin, valueRangeMax],
            discrete: formData.get('discrete') === 'on',
            inertia_weight: parseFloat(formData.get('inertia_weight')),
            cognitive_coefficient: parseFloat(formData.get('cognitive_coefficient')),
            social_coefficient: parseFloat(formData.get('social_coefficient')),
            max_velocity: parseFloat(formData.get('max_velocity')),
            decreasing_inertia: formData.get('decreasing_inertia') === 'on',
            final_inertia_weight: parseFloat(formData.get('final_inertia_weight')),
            use_neighborhood: formData.get('use_neighborhood') === 'on',
            neighborhood_size: parseInt(formData.get('neighborhood_size'))
        };
        
        // Create form data for AJAX request
        const requestData = new FormData();
        requestData.append('config_type', 'pso');
        requestData.append('config_name', configName);
        requestData.append('config_data', JSON.stringify(configData));
        
        // Send AJAX request
        $.ajax({
            url: '/api/save_config',
            type: 'POST',
            data: requestData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.success) {
                    alert('Configuration saved successfully!');
                    // Refresh page to update saved configurations list
                    location.reload();
                } else {
                    alert('Error: ' + response.error);
                }
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                alert('Error: ' + (response.error || 'Failed to save configuration'));
            }
        });
    });
    
    // Load configuration
    $(document).on('click', '.load-config', function() {
        const configType = $(this).data('config-type');
        const configName = $(this).data('config-name');
        
        // Create form data for AJAX request
        const requestData = new FormData();
        requestData.append('config_name', configName);
        
        // Send AJAX request
        $.ajax({
            url: '/api/load_config',
            type: 'POST',
            data: requestData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.success) {
                    // Load configuration into form
                    if (configType === 'ga') {
                        loadGaConfig(response.config_data);
                        $('#ga-tab').tab('show');
                    } else if (configType === 'pso') {
                        loadPsoConfig(response.config_data);
                        $('#pso-tab').tab('show');
                    }
                    
                    alert('Configuration loaded successfully!');
                } else {
                    alert('Error: ' + response.error);
                }
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                alert('Error: ' + (response.error || 'Failed to load configuration'));
            }
        });
    });
    
    // Function to load GA configuration into form
    function loadGaConfig(config) {
        $('#populationSize').val(config.population_size);
        $('#numGenerations').val(config.num_generations);
        $('#chromosomeType').val(config.chromosome_type);
        $('#valueRangeMin').val(config.value_range[0]);
        $('#valueRangeMax').val(config.value_range[1]);
        $('#selectionMethod').val(config.selection_method);
        $('#tournamentSize').val(config.tournament_size);
        $('#crossoverRate').val(config.crossover_rate);
        $('#mutationRate').val(config.mutation_rate);
        $('#elitism').prop('checked', config.elitism);
        $('#eliteSize').val(config.elite_size);
        $('#crossoverType').val(config.crossover_type);
        $('#adaptiveMutation').prop('checked', config.adaptive_mutation);
    }
    
    // Function to load PSO configuration into form
    function loadPsoConfig(config) {
        $('#numParticles').val(config.num_particles);
        $('#numIterations').val(config.num_iterations);
        $('#psoValueRangeMin').val(config.value_range[0]);
        $('#psoValueRangeMax').val(config.value_range[1]);
        $('#discrete').prop('checked', config.discrete);
        $('#inertiaWeight').val(config.inertia_weight);
        $('#cognitiveCoefficient').val(config.cognitive_coefficient);
        $('#socialCoefficient').val(config.social_coefficient);
        $('#maxVelocity').val(config.max_velocity);
        $('#decreasingInertia').prop('checked', config.decreasing_inertia);
        $('#finalInertiaWeight').val(config.final_inertia_weight);
        $('#useNeighborhood').prop('checked', config.use_neighborhood);
        $('#neighborhoodSize').val(config.neighborhood_size);
    }
    
    // Apply GA configuration
    $('#applyGaConfig').on('click', function() {
        // Collect form data
        const formData = new FormData($('#gaForm')[0]);
        
        // Convert value range to array
        const valueRangeMin = parseFloat(formData.get('value_range_min'));
        const valueRangeMax = parseFloat(formData.get('value_range_max'));
        
        // Create config object
        window.gaParams = {
            population_size: parseInt(formData.get('population_size')),
            num_generations: parseInt(formData.get('num_generations')),
            chromosome_type: formData.get('chromosome_type'),
            value_range: [valueRangeMin, valueRangeMax],
            selection_method: formData.get('selection_method'),
            tournament_size: parseInt(formData.get('tournament_size')),
            crossover_rate: parseFloat(formData.get('crossover_rate')),
            mutation_rate: parseFloat(formData.get('mutation_rate')),
            elitism: formData.get('elitism') === 'on',
            elite_size: parseInt(formData.get('elite_size')),
            crossover_type: formData.get('crossover_type'),
            adaptive_mutation: formData.get('adaptive_mutation') === 'on'
        };
        
        // Show success message with toast
        const toastHTML = `
            <div class="toast align-items-center text-white bg-success border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas fa-check-circle me-2"></i> GA configuration applied successfully!
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            </div>
        `;
        
        $('#toastContainer').html(toastHTML);
        const toast = new bootstrap.Toast($('.toast'));
        toast.show();
        
        console.log('GA configuration applied:', window.gaParams);
    });
    
    // Apply PSO configuration
    $('#applyPsoConfig').on('click', function() {
        // Collect form data
        const formData = new FormData($('#psoForm')[0]);
        
        // Convert value range to array
        const valueRangeMin = parseFloat(formData.get('value_range_min'));
        const valueRangeMax = parseFloat(formData.get('value_range_max'));
        
        // Create config object
        window.psoParams = {
            num_particles: parseInt(formData.get('num_particles')),
            num_iterations: parseInt(formData.get('num_iterations')),
            value_range: [valueRangeMin, valueRangeMax],
            discrete: formData.get('discrete') === 'on',
            inertia_weight: parseFloat(formData.get('inertia_weight')),
            cognitive_coefficient: parseFloat(formData.get('cognitive_coefficient')),
            social_coefficient: parseFloat(formData.get('social_coefficient')),
            max_velocity: parseFloat(formData.get('max_velocity')),
            decreasing_inertia: formData.get('decreasing_inertia') === 'on',
            final_inertia_weight: parseFloat(formData.get('final_inertia_weight')),
            use_neighborhood: formData.get('use_neighborhood') === 'on',
            neighborhood_size: parseInt(formData.get('neighborhood_size'))
        };
        
        // Show success message with toast
        const toastHTML = `
            <div class="toast align-items-center text-white bg-success border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas fa-check-circle me-2"></i> PSO configuration applied successfully!
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            </div>
        `;
        
        $('#toastContainer').html(toastHTML);
        const toast = new bootstrap.Toast($('.toast'));
        toast.show();
        
        console.log('PSO configuration applied:', window.psoParams);
    });
});
