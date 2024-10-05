$(document).ready(function(){
    $(document).on('submit', '.no-submit-form', function(e){
        e.preventDefault();
        return false;
    });

    // initialize canvas for handwritten digits
    (function() {
        var canvas = this.__canvas = new fabric.Canvas('digit_canvas', {
            isDrawingMode: true,
            freeDrawingBrush: {
                color: 'rgb(255, 0, 0)',
                width: 20
            }
        });

        fabric.Object.prototype.transparentCorners = false;

        var brushSize = 10
        canvas.setBackgroundColor('rgba(0, 0, 0, 1)', canvas.renderAll.bind(canvas));
        canvas.freeDrawingBrush.color = 'rgba(255, 255, 255, 1)';
        canvas.freeDrawingBrush.width = brushSize;
        canvas.freeDrawingCursor = `url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAABr0lEQVRIibWVPUscURSGX1dtks7CJuDPsApEEMFCbEOiRLKNXbKruBZbiEIWnM0WFmKfH5E+YBVIkSpgYWwsEoLED/AD9ZErZ2D2Mt87vnBh5t475+Gc+547AqoeS8Av4BhoAEPR+FXCxoA2cE+/uk8FfOmBoupWCXwGTAHPLcMkBVUBN4EzoG7vH1Kga4PCmsCNBbsE3th8IwF4MgjMBb3zAl4Bi7a+GmOgP2Vh68BtQhYOumD7PnrQZtkyJsFCXQNvbX/LzngHGC4KW80AReUye2+NP2suLuTSJCNk6VU0TpEyltEuMFEUmNZXaerFxcuCtXIYJE6fnEGKAldKwgL/D5EH2Ipp2ryZpVatSoNs5zGgP9F5SpgbNfXrtyRUTD1J67m/MPIL4J09zwD/q87ML+mkXbqBvc+7X0kGrJvmxixgPRKoY3Pu/jtNgDk31orCQuAo8MUL2LYNc8C5txaUhYXAceAoJost2zQduQCCsqAocCrlnD5HSr43KCwEbmSYY7kKUDhGJL32OuWvpO+Svkn6KemwYF+mygH/STqQ9FXSvqQfNndRJehRkh4AUb1AC50iecgAAAAASUVORK5CYII=') 0 ${brushSize + 12}, crosshair`
        // canvas.freeDrawingBrush.shadow.color = 'rgba(0, 0, 0, 1)';
        // canvas.freeDrawingBrush.shadow.offsetX = 0;
        // canvas.freeDrawingBrush.shadow.offsetY = 0;

        $('#clear_canvas_btn').on('click', function(){
            canvas.clear();
            canvas.setBackgroundColor('rgba(0, 0, 0, 1)', canvas.renderAll.bind(canvas));
        });

        $('#handwritten_predict_btn').on('click', function(e){
            var cnn_task = $(this).data('task');

            var imageBase64 = canvas.toDataURL({
                format: 'png',
                left: 0,
                top: 0,
                width: 200,
                height: 200
            });

            //console.log(imageBase64);

            $('#error_mgs').html('');

            var formData = JSON.stringify({'image_file': imageBase64});
            
            $.ajax({
                url: '/predict_cnn/' + cnn_task,
                type: 'POST',
                data: formData,
                contentType: 'application/json',
                success: function(data) {
                    //console.log(data);
                    $('#result').html(data)
                    show_hide_spinner(false);
                },
                error: function(jqXHR, textStatus, errorThrown) {
                    show_hide_spinner(false);
                    console.log(textStatus, errorThrown);
                    $('#error_mgs').html("Sorry! Something went wrong. (" + errorThrown + ")");
                }
            });
        });

    })();

    $('.btn-prof-category').on('click', function(e){
        var proj_category = $(this).data('profcat');
        //console.log('clicked!', proj_category)

        if(proj_category == 'all') {
            $('.trpf-card').fadeIn(200);
        } else {
            $('.trpf-card').hide();
            $('.' + proj_category).fadeIn(200);
        }
        
    });

    // Upload image for mood detection
    
    function show_hide_spinner(is_show) {
        $('#error_mgs').html('')

        $('.predict-btn').prop('disabled', is_show);
        if(is_show){
            $('#spinner_block').show()            
        } else {
            $('#spinner_block').hide()
        }
    }

    $('#upload_image_btn').on('click', function(e){
        var cnn_task = $(this).data('task');
        
        var image_file = $('#image_file').prop('files');        

        if(image_file.length == 0) {
            $('#error_mgs').html('Please pick an image.')
        } else {
            show_hide_spinner(true);
            var image_obj = image_file[0];
            var image_size = parseFloat(image_obj.size / (1024 * 1024));

            if(image_size > 2) {
                $('#error_mgs').html('Max size exceeded.!' + image_size)
            } else {
                $('#error_mgs').html('')

                encodeImageFileAsURL(image_obj, function(data){
                    $('#image_preview').html('<img src="' + data.result + '" width=100 />')

                    var formData = JSON.stringify({'image_file': data.result});

                    $('#result').html('');

                    $.ajax({
                        url: '/predict_cnn/' + cnn_task,
                        type: 'POST',
                        data: formData,
                        contentType: 'application/json',
                        success: function(data) {
                            //console.log(data);
                            $('#result').html(data)
                            show_hide_spinner(false);
                        },
                        error: function(jqXHR, textStatus, errorThrown) {
                            show_hide_spinner(false);
                            console.log(textStatus, errorThrown);
                            $('#error_mgs').html("Sorry! Something went wrong. (" + errorThrown + ")");
                        }
                    })
                });  
               
            }
        }        
    });

    // Image captioning
    $("#generate_caption_btn").on("click", function(e){
        var image_file = $('#image_file').prop('files');

        if(image_file.length == 0) {
            $('#error_mgs').html('Please pick an image.')
        } else {
            show_hide_spinner(true);
            var image_obj = image_file[0];
            var image_size = parseFloat(image_obj.size / (1024 * 1024));

            if(image_size > 2) {
                $('#error_mgs').html('Max size exceeded.!' + image_size)
            } else {
                $('#error_mgs').html('')

                var formData = new FormData();
                formData.append('image_file', image_obj);

                encodeImageFileAsURL(image_obj, function(data){
                    $('#image_preview').html('<img src="' + data.result + '" width=100 />')
                });                

                $('#result').html('');

                $.ajax({
                    url: '/image_captioning',
                    type: 'POST',
                    cache: false,
                    processData: false,
                    contentType: false,
                    data: formData,
                    enctype: 'multipart/form-data',
                    success: function(data) {
                        //console.log(data);
                        $('#result').html(data)
                        show_hide_spinner(false);
                    },
                    error: function(jqXHR, textStatus, errorThrown) {
                        show_hide_spinner(false);
                        console.log(textStatus, errorThrown);
                        $('#error_mgs').html("Sorry! Something went wrong. (" + errorThrown + ")");
                    }
                })
            }
        }
    });

    // get base64 image data
    function encodeImageFileAsURL(element, callback) {
        var file = element;
        var reader = new FileReader();
        reader.onloadend = function() {
          console.log('RESULT', reader.result)

          callback(reader)
        }
        reader.readAsDataURL(file);
    }

    // send data for banknotes authentication
    $('#validate_banknote_btn').on('click', function(e){
        var variance = $('#variance').val();
        var skewness = $('#skewness').val();
        var curtosis = $('#curtosis').val();
        var entropy  = $('#entropy').val() 

        if(variance == "" || skewness == "" || curtosis == "" || entropy == "") {
            $('#error_mgs').html('All fields are required.')
        } else {
            show_hide_spinner(true);

            var formData = {
                'variance': variance,
                'skewness': skewness,
                'curtosis': curtosis,
                'entropy': entropy
            }
    
            //console.log(formData);
            $.ajax({
                url: '/banknotes_auth',
                type: 'POST',
                data: JSON.stringify(formData),
                contentType: 'application/json',
                success: function(data){
                    //console.log(data);
                    $('#result').html(data)
                    show_hide_spinner(false);
                },
                error: function(jqXHR, textStatus, errorThrown) {
                    show_hide_spinner(false);
                    console.log(textStatus, errorThrown);
                    $('#error_mgs').html("Sorry! Something went wrong. (" + errorThrown + ")");
                }
            });
        }
    });

    // send data for black friday purchase prediction
    $('#predict_black_friday_btn').on('click', function(e){

        var formData = {
            'gender': $("input[name='gender']:checked").val(),
            'age': $('#age').val(),
            'occupation': $('#occupation').val(),
            'stay_in_current_city': $('#stay_in_city').val(),
            'marital_status': $("input[name='marital_status']:checked").val(),
            'product_main_category': $('#product_main_category').val(),
            'product_category_1': $('#product_sub_category_1').val(),
            'product_category_2': $('#product_sub_category_2').val(),
            'city_category': $('#city_category').val()
        }

        // console.log(formData)
        show_hide_spinner(true);

        $.ajax({
            url: '/black_friday_prediction',
            type: 'POST',
            data: JSON.stringify(formData),
            contentType: 'application/json',
            success: function(data) {
                //console.log(data)
                $('#result').html(data)
                show_hide_spinner(false);
            },
            error: function(jqXHR, textStatus, errorThrown) {
                show_hide_spinner(false);
                console.log(textStatus, errorThrown);
                $('#error_mgs').html("Sorry! Something went wrong. (" + errorThrown + ")");
            }
        });

    });

    // Recomender system
    $('#recommend_movies').on('click', function(e){
        var input_movie = $('#your_movie').val()

        if(input_movie == "") {
            $('#error_mgs').html('Please enter a movie title.')
        } else {
            show_hide_spinner(true);

            $.ajax({
                url: '/recommender_content_based',
                type: 'POST',
                data: JSON.stringify({
                    'input_movie': input_movie
                }),
                contentType: 'application/json',
                success: function(data) {
                    //console.log(data)
                    $('#result').html(data)
                    show_hide_spinner(false);
                },
                error: function(jqXHR, textStatus, errorThrown) {
                    show_hide_spinner(false);
                    console.log(textStatus, errorThrown);
                    $('#error_mgs').html("Sorry! Something went wrong. (" + errorThrown + ")");
                }
            });
        }        
    });
    

    function reviver( key, value ) {
        if ( value === "NaN" ) {
            return NaN;
        }
        if ( value === "nan" ) {
            return NaN;
        }
        if ( value === "***Infinity***" ) {
            return Infinity;
        }
        if ( value === "***-Infinity***" ) {
            return -Infinity;
        }
        return value;
    }

    // CO2 emission prediction
    $('#year_range').on('input', function(e){
        var range_val = $(this).val();
        console.log(range_val)
        $('#range_text').html(range_val)
    });

    var co2_chart;
    $('#predict_co2_emission_btn').on('click', function(e){
        var to_year =  $('#year_range').val();
        const ctx = document.getElementById("chart_block");

        if(co2_chart) {
            co2_chart.destroy();
        }

        show_hide_spinner(true);

        $.ajax({
            url: '/co2_emission_lstm',
            type: 'POST',
            data: JSON.stringify({
                'to_year':to_year
            }),
            contentType: 'application/json',
            success: function(data) {
                //console.log(data);
                //$('#result').html(data);

                //var data = JSON.parse( responseData, reviver );

                show_hide_spinner(false);
                //data = JSON.parse(data)
                //console.log(data['chart_data'])
                if(data != undefined) {
                    
                    co2_chart = new Chart(ctx, {
                        type: "line",
                        data: {
                          labels: data['x_labels'],
                          datasets: [
                            {
                                data:  data['orig'].map(item => item == 0 ? NaN : item),
                                label: "CO2 emission",
                                borderColor: 'rgba(0, 150, 255, 1)',
                                fill: false
                            },

                            {
                                data:  data['predicted'].map(item => item == 0 ? NaN : item),
                                label: "CO2 emission prediction",
                                borderColor: 'rgba(199, 0, 57, 1)',
                                fill: false
                            },
                            
                          ]
                        },
                        options: {
                            responsive: true,
                            title: {
                                display: true,
                                text: "CO2 emission in Sri Lanka (in kilotons)"
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Year'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'CO2 emission (in kilotons)'
                                    }
                                }
                            }
                        }
                      });
                }
                
            },
            error: function(jqXHR, textStatus, errorThrown) {
                show_hide_spinner(false);
                console.log(textStatus, errorThrown);
                $('#error_mgs').html("Sorry! Something went wrong. (" + errorThrown + ")");
            }
        })
    });

    // Heart attack prediction
    $('#predict_heart_attack_btn').on('click', function(e){
        
        var age = $('#age').val()
        var gender = $('input[name=gender]:checked').val()
        var chest_pain_type = $('#chest_pain_type').val()
        var resting_blood_pressure = $('#resting_blood_pressure').val()
        var cholesterol_level = $('#cholesterol_level').val()
        var fbs = $('input[name=fbs]:checked').val()
        var resting_electrocardio = $('#resting_electrocardio').val()
        var max_heart_rate = $('#max_heart_rate').val()
        var exang = $('input[name=exang]:checked').val()
        var oldpeak = $('#oldpeak').val()
        var slp = $('#slp').val()
        var num_major_vessels = $('#num_major_vessels').val()
        var thall = $('#thall').val()

        var form_data = {
            'age': age,
            'sex': gender,
            'cp': chest_pain_type,
            'trtbps': resting_blood_pressure,
            'chol': cholesterol_level,
            'fbs': fbs,
            'restecg': resting_electrocardio,
            'thalachh': max_heart_rate,
            'exang': exang,
            'oldpeak': oldpeak,
            'slp': slp,
            'caa': num_major_vessels,
            'thall': thall
        }

        if(resting_blood_pressure == "" || cholesterol_level == "" || max_heart_rate == "" || oldpeak == "") {
            show_hide_spinner(false);
            $('#error_mgs').html('All fields are required.')
        } else {
            show_hide_spinner(true);

            $.ajax({
                url: '/heart_attack_prediction',
                type: 'POST',
                data: JSON.stringify(form_data),
                contentType: 'application/json',
                success: function(data) {
                    //console.log(data);
                    $('#result').html(data)
                    show_hide_spinner(false);
                },
                error: function(jqXHR, textStatus, errorThrown) {
                    show_hide_spinner(false);
                    console.log(textStatus, errorThrown);
                    $('#error_mgs').html("Sorry! Something went wrong. (" + errorThrown + ")");
                }
            });
        }
    });

    // Diabetes prediction
    $('#predict_diabetes_btn').on('click', function(e){
        var form_data = Object.fromEntries(new URLSearchParams($('#model_values_form').serialize()))
        
        show_hide_spinner(true);
        $.ajax({
            url: '/diabetes_risk_prediction',
            type: 'POST',
            data: JSON.stringify(form_data),
            contentType: 'application/json',
            success: function(data) {
                $('#result').html(data)
                show_hide_spinner(false);
            },
            error: function(jqXHR, textStatus, errorThrown) {
                show_hide_spinner(false);
                console.log(textStatus, errorThrown);
                $('#error_mgs').html("Sorry! Something went wrong. (" + errorThrown + ")");
            }
        });
    });

});

$(window).on('load', function () {
    if(window.location.pathname == '/') {
        $('#projectCategoriesButtons').show();

        $('#countAllProjects').html($('.trpf-card').length);
        $('#countMLProjects').html($('.ml').length);
        $('#countDLProjects').html($('.dl').length);
        $('#countAnalysisProjects').html($('.dashboard').length);
        $('#countVolunteerProjects').html($('.volunteer').length);
    } else {
        $('#projectCategoriesButtons').hide();
    }
});