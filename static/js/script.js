$(document).ready(function(){
    $(document).on('submit', '.no-submit-form', function(e){
        e.preventDefault();
        return false;
    });

    $('#upload_image_btn').on('click', function(e){
        var image_file = $('#image_file').prop('files');

        if(image_file.length == 0) {
            $('#error_mgs').html('Please pick an image.')
        } else {
            var image_obj = image_file[0];
            var image_size = parseFloat(image_obj.size / (1024 * 1024));

            if(image_size > 1) {
                $('#error_mgs').html('Max size exceeded.!' + image_size)
            } else {
                $('#error_mgs').html('')

                var formData = new FormData();
                formData.append('image_file', image_obj);

                $.ajax({
                    url: '/predict_mood',
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
});