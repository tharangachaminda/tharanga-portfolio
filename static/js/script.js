$(document).ready(function(){
    $(document).on('submit', '.no-submit-form', function(e){
        e.preventDefault();
        return false;
    });

    // Upload image for mood detection
    $('.predict-btn').on('click', function(e){
        $('#result').html('Processing...')
    });

    $('#upload_image_btn').on('click', function(e){
        var cnn_task = $(this).data('task');
        
        var image_file = $('#image_file').prop('files');

        if(image_file.length == 0) {
            $('#error_mgs').html('Please pick an image.')
        } else {
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
                    url: '/predict_cnn/' + cnn_task,
                    type: 'POST',
                    cache: false,
                    processData: false,
                    contentType: false,
                    data: formData,
                    enctype: 'multipart/form-data',
                    success: function(data) {
                        console.log(data);
                        $('#result').html(data)
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
        
        var formData = {
            'variance': $('#variance').val(),
            'skewness': $('#skewness').val(),
            'curtosis': $('#curtosis').val(),
            'entropy': $('#entropy').val()
        }

        //console.log(formData);
        $.ajax({
            url: '/banknotes_auth',
            type: 'POST',
            data: JSON.stringify(formData),
            contentType: 'application/json',
            success: function(data){
                console.log(data);
                $('#result').html(data)
            }
        });
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

        $.ajax({
            url: '/black_friday_prediction',
            type: 'POST',
            data: JSON.stringify(formData),
            contentType: 'application/json',
            success: function(data) {
                console.log(data)
                $('#result').html(data)
            }
        });

    });

    // Recomender system
    $('#recommend_movies').on('click', function(e){
        var input_movie = $('#your_movie').val()

        if(input_movie == "") {
            $('#error_mgs').html('Please enter a movie title.')
        } else {
            $.ajax({
                url: '/recommender_content_based',
                type: 'POST',
                data: JSON.stringify({
                    'input_movie': input_movie
                }),
                contentType: 'application/json',
                success: function(data) {
                    console.log(data)
                    $('#result').html(data)
                }
            });
        }        
    });
    
});