$(document).ready(function() {
    let bookNames = [];

    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    $.get('/book_names', function(data) {
        if (Array.isArray(data)) {
            bookNames = data;
            $('#book-count').text(data.length); // Cập nhật số lượng sách
        } else if (data.error) {
            console.error('Error loading book names:', data.error);
        }
    }).fail(function(xhr, status, error) {
        console.error('AJAX error:', error);
    });

    const showSuggestions = debounce(function(input) {
        $('#book-list').empty();
        if (!input) {
            $('.suggestions').remove();
            return;
        }
        const suggestions = bookNames.filter(name => name.toLowerCase().includes(input.toLowerCase()));
        if (suggestions.length > 0) {
            if (!$('.suggestions').length) {
                $('<div class="suggestions">').appendTo('.search-bar');
            } else {
                $('.suggestions').empty();
            }
            suggestions.forEach(suggestion => {
                $('<div class="suggestion-item">')
                    .text(suggestion)
                    .on('click', function() {
                        $('#book-search').val(suggestion);
                        $('.suggestions').remove();
                    })
                    .appendTo('.suggestions');
            });
        } else {
            $('.suggestions').remove();
        }
    }, 300);

    $('#book-search').on('input', function() {
        const input = $(this).val();
        showSuggestions(input);
    });

    $(document).click(function(e) {
        if (!$(e.target).closest('.search-bar').length) {
            $('.suggestions').remove();
        }
    });

    $('#search-btn').click(function() {
        var bookName = $('#book-search').val();
        if (bookName.trim() === "") {
            alert("Please enter a book name!");
            return;
        }

        $.ajax({
            url: '/get_book_info',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({book_name: bookName}),
            success: function(bookInfo) {
                $('#selected-book').empty();
                if (bookInfo && bookInfo['image_book']) {
                    var selectedCard = `
                        <div class="book-card" data-link="${bookInfo['url']}">
                            <img src="${bookInfo['image_book']}" alt="${bookInfo['title']}" class="book-img">
                            <div class="book-info">
                                <h3 class="book-title">${bookInfo['title']}</h3>
                            </div>
                        </div>
                    `;
                    var $selectedCard = $(selectedCard);
                    $selectedCard.on('click', function() {
                        window.open(bookInfo['url'], '_blank');
                    });
                    $('#selected-book').append($selectedCard);
                } else {
                    $('#selected-book').html('<p>Book not found.</p>');
                }
            },
            error: function() {
                $('#selected-book').html('<p>Unable to load book info.</p>');
            }
        });

        $.ajax({
            url: '/recommend',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({book_name: bookName}),
            success: function(data) {
                $('#book-list').empty();
                if (data.length > 0) {
                    $('#book-list').append('<h3 class="text-center mb-4" style="color: #00ff00;">We recommend some books like this...</h3>');
                    data.forEach(function(book) {
                        var card = `
                            <div class="col-md-4 mb-4">
                                <div class="book-card" data-link="${book['url']}">
                                    <div class="img-container">
                                        <img src="${book['image_book']}" alt="${book['title']}" class="book-img">
                                        ${book['video'] && book['video'] !== '' ? `
                                            <div class="video-container">
                                                <video class="trailer-video" muted>
                                                    <source src="${book['video']}" type="video/mp4">
                                                    Your browser does not support the video tag.
                                                </video>
                                            </div>
                                        ` : ''}
                                    </div>
                                    <div class="book-info">
                                        <h3 class="book-title">${book['title']}</h3>
                                        <p class="book-desc">${book['description']}</p>
                                        <p class="book-genre">Genre: ${book['genres']}</p>
                                    </div>
                                </div>
                            </div>
                        `;
                        var $card = $(card);
                        
                        $card.find('.book-card').on('click', function() {
                            window.open(book['url'], '_blank');
                        });
                        $('#book-list').append($card);
                    });
                } else {
                    $('#book-list').append('<p class="text-center">No similar books found.</p>');
                }
            },
            error: function() {
                alert("An error occurred, please try again!");
            }
        });
    });

    $('#random-btn').click(function() {
        $.get('/random_book', function(bookInfo) {
            if (bookInfo && bookInfo['title']) {
                $('#book-search').val(bookInfo['title']);
                $('#search-btn').click(); // Trigger search with random book
            } else {
                alert("Unable to fetch a random book.");
            }
        }).fail(function() {
            alert("Error fetching random book!");
        });
    });
});