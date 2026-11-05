"""Golden trellis decode vectors from the research encoder.

Each vector is one Gaussian block encoded by the piper-research trellis encoder
(``encode_rows_fast`` in ``projects/e8-w4a8-vq/trellis_w4.py``, research commit c25eaf6),
packed by its ``pack`` and checked against its own ``decode_stream`` before export.
They pin this package's decode to the format the research checkpoints use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchVector:
    """One encoded block: layout, FP8 E4M3 scale bits, stream and expected INT8 values."""

    block_size: int
    block_bits: int
    scale_code: int
    bitstream_hex: str
    values_hex: str


RESEARCH_VECTORS = (
    ResearchVector(
        256,
        768,
        0x5C,
        (
            "6a4c18b475f9876b4adab613a380df2ed83a9f771410788a2d42750caeb0de1ac24079085b0d907d"
            "fcc272053d5d1153e19b4137a71468c8d526e1f24ecf327cd363d642a5c1568e7b513a44c9cf706f"
            "88f5df04b15a21be30d47c11b3c33f78"
        ),
        (
            "ff08f4060618fd0efbf9051f1908e72c2d1326e70c15c5fb09f1d1ef18e50cfb18080afedd03f3f0"
            "eef5f4fed6fb0c10150e15fbfedb0dfdf313e5f714ebf9e813e40ef73b1f0bddf2430ffff5070307"
            "f9ffe6041b0fef150121d703e7ee1f01e8d9e307dce51012eaee05e8f7e5321af91000f6e812c9ee"
            "23fdda03d520190610df19fa130f0de827e6e2fb061a0024070915ea15fef7cffb18ebfc0208fb03"
            "22eceaf5f8ebfbe61116ff0be511f1e4ed13d7ff00f81024141bfdd6ef0c06eb04eafeebfcf41dec"
            "03f4e50312f4ed12cb0a18fcf9f7f8fe0b22fe0ff4e7ee0a1f071dfd32e418e51d020efb0bf011e9"
            "01f80100fbe2011b120434f91312f5f5"
        ),
    ),
    ResearchVector(
        256,
        888,
        0x62,
        (
            "c9b91d21c9826b21d00957f46c947157da04d3504d2392b6ba1bc61470c4bd04b0aa23fee321db40"
            "9c966c3f02225db146df8e53ad3839ef2113f1025c89faed83521c4428e6ded0d18edcf6e187c8d1"
            "ca2b7dfd71cef3c34c3726577751f6a40e5fe9d84f2b242731f63b73b3ae78"
        ),
        (
            "01fb08effaecb0ff07c61b8b22bbf2d50509e3fbe11508ef1e010dd402bb25ed09e1f0f22c2debf6"
            "f901e7fc081eda1220380f3ff3f6f8f9dcfa21d83bebff022005fcf31a08f1f4f5fff217b81d0ed5"
            "120a13e91c1227d10ff71b1e57f3b442fe16f115eb0ef941ed0f08060f1436e237f3f40708ec1d13"
            "ead704d8fbe9fa1cd0d3c6ce1d05223933fe0cfe13130636f4eddc0df8f3fc362d1e29feed480d27"
            "e5e8eefa22bde3f8e51d0d33d9ffda39ea291d05dfe30722e83ff1c9fbebef4725c516fa03e405cd"
            "11e62a01e72f011c203ff63631ddc9f516d00726fc09f5fffae239320c04ffd60bf4cb28e00bfccf"
            "da09cde4ee19fc17f214061818360c0f"
        ),
    ),
    ResearchVector(
        256,
        1024,
        0x61,
        (
            "f39bbd54cdb0170c1ad8c9d57ac45b87806c031e18bd5af7fed98433dc4eee0289dae5e071e527c7"
            "9932f02303c8b472ef5a90918738c225b8cc2b206cbd38ed8bd025461a9af5be6f4240fde629718a"
            "bd6425ec380352d7119005a86ce917656bdcbdbfa909391f9817660e4d4af5875bb5ae1a225b3ff4"
            "c5ef7f1c35822f35"
        ),
        (
            "e40b53df052c2e0ff2cce0131dd711123bf213cfee18b021d923170f26e345dfeee6fe1909dcfd23"
            "0217e218ccc90915fbddc4de00d8ed1032e30f38f3eae105060ddf00f4ed07040bf5e5fa38f9381a"
            "e8200fe9cb0e052d325724dbccedf4e205d8e320100606d7bb1627f33efaf632e903f0eef8f8e20e"
            "1ceddac4fb30151b0bdf0ef624faec08080de02018d7b3f5ef13350ce7f6f3c54f03f3fcdcef1900"
            "bce4d10112ebd13dfde4e613adeec618ea04f232e809fee33de7eb022f084c17eae355e7eceef8e3"
            "20f000fd3617f4eefc1500210d01beee310ac1f3d0261004e803f1faef04dc25faea18eb4b1bf515"
            "0006fb0215ea1bdc27f1e64ef5ceebf8"
        ),
    ),
    ResearchVector(
        256,
        1144,
        0x63,
        (
            "dcf15a10e5bf8c7469100e3d20cb695c512dfc988abeb3efe92081c0143c50a01e16e481769c13e8"
            "60e37bab3f28a3f515bafa7ed5d2a9590f3246d0a83dbaace0e0d8ca81ad3222aca7e6c16bd6a1e5"
            "d5e547f4f8c94facec1d6f9840c5c6a2311d13ee2568e73a29f8726093443076a9ec9477e1c3e53c"
            "28d1e70c3fa9ee684046fea4aaa5670403a2459ab20c83"
        ),
        (
            "f1c12d011bd514b231fbfe180ff3f3dd1f2b08dc03f7cde8012b412c03efb4245babd5e402e0e207"
            "d4ee473e204e23f81cf03af8cafef2c3e01d2af8fbc6e02ceb2423080c33dded17191dfb08f842f0"
            "4dfcb7e4bcef97cc35130f0ff03af304113ec2edbb0ee0331eeaf1e04c06e021f6b604450b021ff3"
            "406e260f38ebf3151700ef78fd18eecbc4cecce6e312ef15f0c40652041ce0ee1a17e825dce1fd16"
            "61d2d5ecaa1b30f20be2d7d5f2094c032ae9e9360af7b4f4ae260d1af8e8c103eee13c25d62ed7fb"
            "4dd8421dfdf30efc32380b022cebf600e501f820e80d142807d42beb0104a110de13d3e9dd110548"
            "0435f13a15ec49053ad6ead9c5e012f8"
        ),
    ),
    ResearchVector(
        224,
        1000,
        0x63,
        (
            "b0341a28510e47da93ca783d3277ff422d1b65f2761b07c599df44f13cb0d5a808082f1b5acfcc1f"
            "01eeb468cb070b280bb8b9b5614fb91f3ee0612c404d5458d18a8ae9c40ca3ae81bc2b12e6fb056f"
            "edadd5df4a5f01fb90b07a48ad3ae4da17533ab3e6e5082e7180a5077d86237b96489e57e2e1704e"
            "e45866cc17"
        ),
        (
            "2027ee06f3fe513edbdadf31d23407e7e0122f96d31f0745124ff31911001c2e00172326ee19e7de"
            "0acd37b4fcff37de04ecaa19fae049c8fbc4e3e9ffe21cb8e053081df61834333f08c928eaf7ed18"
            "cac9d70d3518f33af9dbd3dd10fafbf1eed82fff0c09e6b2c61b12280eebddedfde3b500e8c9ffb8"
            "b7e636273dec2e171006fae124fff51bee03dbf5faa5fff9f0ece4f613e349203a10d235eef9f630"
            "31ff1a173252f93cfebf09f6d6dd3133f92ec33841bff6ff362018f4ca3708d6de04ad2105d3c72e"
            "360ad006f3fa22d41292eb5301dffbb8ca01fc2ef6e8d921"
        ),
    ),
    ResearchVector(
        256,
        1400,
        0x5E,
        (
            "acb17c2220a7ef1f01a9c31dcec7d6ac500036f8238c545214672bcd9917ada8eba345a6a8e91be1"
            "b84df24ef26a4c469147b4cc7a4f23a89d693f7e2d09674581460a4e9f60eebb0e7820f2cd167061"
            "79e0bfe3e02cff0eca6bf8b39bd2f0667c3ed69fd5fda59e82f01c4c33a051b7cec8564f5036a355"
            "516fd8003db4a25773af1a85d265cd386b7a74324c36d99a79979988339f88a3c2cb5b4f9f0b37a4"
            "323a0bf508ee0addfb9839721ed441"
        ),
        (
            "133a0c04efe33af6d3e5fe001d06cb19e1f310e702eb09fbd511f3d2f4eb11ff26d60ff6150a0c04"
            "04d71503f1f01d24ccf8f808f4d9f801004007f1ce05e806ec2b2010f201fc10f104f9e01322d91b"
            "fa0af1ff09f135de13edf229f0ead8f9e1eee0190014ea000d0826dffc0df0050cd1f20e0ef8ef14"
            "24fd21ce2b1bfe20ebfdf722f7fbd22b0cfbf5191f071b04eff4df130d111106012322fd00dc1022"
            "e6261d150116fef817ef0bfbf00a0fe1fc1edffcfaf7eaf7d8def5b710d5200dd206051fd30007d9"
            "081cc80a0ce811d30502e7cd010919fcd8ee1c05e3e811f4180be6faceec1b11f444220c0514eb05"
            "020124c9f335fad72103effc05f91415"
        ),
    ),
    ResearchVector(
        256,
        1536,
        0x68,
        (
            "cbe520edd092bbcbda527f8a3024a927654273a3b6b65908397717d36c8ec733430ccb7817c848f4"
            "4f6651b09e2a322700a926d3a4b642cfe0f1751c2c300e62119e17ad0469b2230042e258badd53d9"
            "e7b412c08b659aaa6add765d8f213cb2b1bf813f95ebc8893de2324fc4b34aa026ed4e0bf0a00593"
            "52f488309a23f978d161ca19bfef7c66896ab89d044a0d837d88674a22e1346c0df45d9c310e5e9b"
            "e77e03351667a51f8bf227ec1e72f4b21a0a003c565550a4c3d79fb78f64b111"
        ),
        (
            "00094126ee4fd9cf9a09c394d60b4432ff2d225432420346f0443941dae5f632f78722dbf6b2fdd6"
            "c8a8e81cd5f932eac1ce01b942d1242d0cd22d17fac067eddf329ee721f7cd0a06ffd4de6ee62d1e"
            "67e620ff6f30f2cafda40df60fe8d6f407ffe62c0d4c1aab299d1ad0a9b2bac940f03a17fa49e01a"
            "ebf20ce2ae08c9b8f43a13fb81bffe84217229f8c9ebf211bf4bfc43290a2fccfad6eee5342fcc03"
            "d40dae0917d3590dd0c310cb03ded25fd5dae581d71df147bcef47e7c93b2a0d4ff70111cbe333e9"
            "0f1a93fefa1a7a066af3fc04bb1629c9d9152d48d01024dab3241fa10317f9321f3713f4d3340ab2"
            "c8ed24dfeffd2411ec2af122e34f64f9"
        ),
    ),
)
